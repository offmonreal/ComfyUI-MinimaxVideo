"""MiniMax H3 Video (V2) ComfyUI node.

Targets the V2 ``video_generation`` API. PAYG access only. Uses the V2
``content`` array request shape and the V2 ``/v2/query/video_generation/{task_id}``
polling endpoint. On success, the generated video URL is read from
``task.content.url`` (NOT from a separate file retrieval call).

All generation-mode / resolution / duration / aspect-ratio combinations are
validated locally before any HTTP request is sent.
"""

import time
from typing import Any, Dict, List, Tuple

from minimax_common import (
    _get_api_key,
    download_file,
    http_json_request,
    make_output_path,
    normalize_base_url,
    pil_to_data_url_jpeg,
    sleep_poll,
    tensor_to_pil_rgb,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V2_DEFAULT_HOST = "api.minimax.io"
V2_PATH_VIDEO_GENERATION = "/v2/video_generation"
V2_PATH_QUERY_TEMPLATE = "/v2/query/video_generation/{task_id}"

# V2 currently exposes a single model; intentionally not a dropdown.
V2_MODEL = "MiniMax-H3"

V2_RESOLUTIONS = ["768P", "2K"]

# V2 supports integer durations from 4 through 15 seconds inclusive.
V2_DURATIONS: List[int] = list(range(4, 16))  # 4..15

V2_ASPECT_RATIOS = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]

# V2-documented polling statuses (lowercase).
V2_STATUS_QUEUED = "queued"
V2_STATUS_RUNNING = "running"
V2_STATUS_SUCCEEDED = "succeeded"
V2_STATUS_FAILED = "failed"
V2_STATUS_CANCELLED = "cancelled"
V2_KNOWN_POLLING_STATUSES = {V2_STATUS_QUEUED, V2_STATUS_RUNNING}


# ---------------------------------------------------------------------------
# Local validation
# ---------------------------------------------------------------------------

def _validate_duration(duration: int) -> None:
    if duration not in V2_DURATIONS:
        raise ValueError(
            f"Duration '{duration}s' is not allowed for V2 H3. "
            f"Allowed: {', '.join(str(d) + 's' for d in V2_DURATIONS)}."
        )


def _validate_resolution(resolution: str) -> None:
    if resolution not in V2_RESOLUTIONS:
        raise ValueError(
            f"Resolution '{resolution}' is not allowed for V2 H3. "
            f"Allowed: {', '.join(V2_RESOLUTIONS)}."
        )


def _validate_aspect_ratio_for_mode(mode: str, aspect_ratio: str) -> None:
    if aspect_ratio not in V2_ASPECT_RATIOS:
        raise ValueError(
            f"Aspect ratio '{aspect_ratio}' is not supported. "
            f"Allowed: {', '.join(V2_ASPECT_RATIOS)}."
        )
    if mode == "📝Text-to-Video":
        if aspect_ratio == "adaptive":
            raise ValueError(
                "Text-to-Video requires an explicit aspect ratio. "
                "'adaptive' is only valid for image-conditioned modes."
            )
    elif mode in (
        "🖼️First Frame Image-to-Video",
        "🖼️Last Frame Image-to-Video",
        "🧷First + Last Frame",
    ):
        # Aspect ratio is determined by the image; adaptive is the only
        # value that makes sense.
        if aspect_ratio != "adaptive":
            raise ValueError(
                f"{mode} derives its aspect ratio from the input image. "
                "Use aspect ratio 'adaptive'."
            )
    elif mode == "🖼️Reference Image-to-Video":
        # adaptive is allowed; explicit supported ratios are also allowed.
        # (already validated above by membership in V2_ASPECT_RATIOS)
        return
    else:
        raise ValueError(f"Unknown generation mode: '{mode}'.")


# ---------------------------------------------------------------------------
# Content array construction
# ---------------------------------------------------------------------------

def _build_content_items(
    mode: str,
    prompt: str,
    first_frame_image,
    last_frame_image,
    reference_image,
) -> List[Dict[str, Any]]:
    """Assemble the V2 ``content`` array for the given mode.

    Every request must contain exactly one non-empty text item, plus zero or
    more image items as required by the mode. ``role`` discriminates the
    image semantic so the API knows how to interpret it.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string for V2 H3 generation.")

    items: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    if mode == "📝Text-to-Video":
        pass

    elif mode == "🖼️First Frame Image-to-Video":
        if first_frame_image is None:
            raise ValueError("First Frame Image-to-Video requires 🖼️First Frame Image.")
        items.append({
            "type": "image_url",
            "image_url": {"url": pil_to_data_url_jpeg(tensor_to_pil_rgb(first_frame_image))},
            "role": "first_frame",
        })

    elif mode == "🖼️Last Frame Image-to-Video":
        if last_frame_image is None:
            raise ValueError("Last Frame Image-to-Video requires 🖼️Last Frame Image.")
        items.append({
            "type": "image_url",
            "image_url": {"url": pil_to_data_url_jpeg(tensor_to_pil_rgb(last_frame_image))},
            "role": "last_frame",
        })

    elif mode == "🧷First + Last Frame":
        if first_frame_image is None:
            raise ValueError("First + Last Frame requires 🖼️First Frame Image.")
        if last_frame_image is None:
            raise ValueError("First + Last Frame requires 🖼️Last Frame Image.")
        items.append({
            "type": "image_url",
            "image_url": {"url": pil_to_data_url_jpeg(tensor_to_pil_rgb(first_frame_image))},
            "role": "first_frame",
        })
        items.append({
            "type": "image_url",
            "image_url": {"url": pil_to_data_url_jpeg(tensor_to_pil_rgb(last_frame_image))},
            "role": "last_frame",
        })

    elif mode == "🖼️Reference Image-to-Video":
        if reference_image is None:
            raise ValueError(
                "Reference Image-to-Video requires 🖼️Reference Image (for Image-to-Video)."
            )
        items.append({
            "type": "image_url",
            "image_url": {"url": pil_to_data_url_jpeg(tensor_to_pil_rgb(reference_image))},
            "role": "reference_image",
        })

    else:
        raise ValueError(f"Unknown generation mode: '{mode}'.")

    return items


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class MinimaxH3VideoGenerate:
    """V2 H3 Video node.

    Uses the V2 ``content`` array request shape and the V2 polling endpoint.
    Does NOT advertise Token Plan support — V2 currently requires PAYG.
    """

    CATEGORY = "Ricksf-Toolbox"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("📁Video Path", "🔗Video URL", "🆔Task ID", "📌Status", "🎞️Video")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🎬Generation Mode": (
                    [
                        "📝Text-to-Video",
                        "🖼️First Frame Image-to-Video",
                        "🖼️Last Frame Image-to-Video",
                        "🧷First + Last Frame",
                        "🖼️Reference Image-to-Video",
                    ],
                ),
                "📝Prompt": ("STRING", {"multiline": True, "default": ""}),
                "🖼️Resolution": (V2_RESOLUTIONS, {"default": "768P"}),
                "⏱️Duration (seconds)": ([str(d) for d in V2_DURATIONS], {"default": "6"}),
                "📐Aspect Ratio": (V2_ASPECT_RATIOS, {"default": "adaptive"}),
                "🌐Base URL": ("STRING", {"default": f"https://{V2_DEFAULT_HOST}"}),
                "🔑API Key (PAYG)": ("STRING", {"default": ""}),
                "⌛Max Wait (seconds)": ("INT", {"default": 1200, "min": 10, "max": 3600}),
                "🔁Poll Interval (seconds)": ("INT", {"default": 10, "min": 5, "max": 60}),
            },
            "optional": {
                "🖼️Reference Image (for Image-to-Video)": ("IMAGE",),
                "🖼️First Frame Image": ("IMAGE",),
                "🖼️Last Frame Image": ("IMAGE",),
            },
        }

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def generate(self, **kwargs) -> Tuple[str, str, str, str, Any]:
        mode = str(kwargs.get("🎬Generation Mode", "")).strip()
        prompt = str(kwargs.get("📝Prompt", "") or "")
        resolution = str(kwargs.get("🖼️Resolution", "768P"))
        duration = int(kwargs.get("⏱️Duration (seconds)", "6"))
        aspect_ratio = str(kwargs.get("📐Aspect Ratio", "adaptive"))
        base_url_raw = str(kwargs.get("🌐Base URL", f"https://{V2_DEFAULT_HOST}"))
        api_key_widget = str(kwargs.get("🔑API Key (PAYG)", "") or "")
        max_wait_s = int(kwargs.get("⌛Max Wait (seconds)", 1200))
        poll_interval_s = int(kwargs.get("🔁Poll Interval (seconds)", 10))
        reference_image = kwargs.get("🖼️Reference Image (for Image-to-Video)")
        first_frame_image = kwargs.get("🖼️First Frame Image")
        last_frame_image = kwargs.get("🖼️Last Frame Image")

        # ---- Authentication -------------------------------------------------
        api_key = _get_api_key(api_key_widget)
        if not api_key:
            raise ValueError(
                "No API key provided. Enter the API Key (PAYG) on the node, "
                "set the MINIMAX_API_KEY environment variable, or configure it in config.json."
            )

        base_url = normalize_base_url(base_url_raw, V2_DEFAULT_HOST)

        # ---- Local validation ----------------------------------------------
        _validate_duration(duration)
        _validate_resolution(resolution)
        _validate_aspect_ratio_for_mode(mode, aspect_ratio)

        # Reference Image-to-Video and First/Last Frame are mutually exclusive
        # because the V2 API does not document a combined reference+frame mode
        # and sending both would produce an ambiguous payload.
        if mode in ("🖼️Reference Image-to-Video",) and (
            first_frame_image is not None or last_frame_image is not None
        ):
            raise ValueError(
                "Reference Image-to-Video is mutually exclusive with "
                "First + Last Frame. Disconnect 🖼️First Frame Image and "
                "🖼️Last Frame Image when using reference mode."
            )
        if mode in ("🧷First + Last Frame",) and reference_image is not None:
            raise ValueError(
                "First + Last Frame is mutually exclusive with "
                "Reference Image-to-Video. Disconnect 🖼️Reference Image "
                "(for Image-to-Video) when using frame mode."
            )

        # ---- Build V2 content array ----------------------------------------
        content_items = _build_content_items(
            mode=mode,
            prompt=prompt,
            first_frame_image=first_frame_image,
            last_frame_image=last_frame_image,
            reference_image=reference_image,
        )

        payload: Dict[str, Any] = {
            "model": V2_MODEL,
            "content": content_items,
            "resolution": resolution,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        # ---- 1. Create the V2 task -----------------------------------------
        create_url = f"{base_url}{V2_PATH_VIDEO_GENERATION}"
        create_resp = http_json_request(
            method="POST",
            url=create_url,
            api_key=api_key,
            payload=payload,
        )
        task_id = str(create_resp.get("task_id") or "")
        if not task_id:
            raise ValueError(f"Failed to create V2 H3 task, API returned: {create_resp}")

        # ---- 2. Poll V2 task status ----------------------------------------
        start = time.time()
        status = ""
        video_url = ""
        while True:
            get_url = f"{base_url}{V2_PATH_QUERY_TEMPLATE.format(task_id=task_id)}"
            task_resp = http_json_request(method="GET", url=get_url, api_key=api_key)
            status = str(task_resp.get("status") or "")

            if status == V2_STATUS_SUCCEEDED:
                # URL lives at task.content.url per V2 documentation.
                task_content = task_resp.get("content") or {}
                video_url = str(task_content.get("url") or "")
                if not video_url:
                    raise ValueError(
                        "V2 H3 task succeeded but no content.url was returned."
                    )
                break

            if status == V2_STATUS_FAILED:
                error_msg = task_resp.get("error_message", "Unknown error")
                raise ValueError(f"Video generation failed: {error_msg}")

            if status == V2_STATUS_CANCELLED:
                raise ValueError("Video generation was cancelled by the server.")

            if status not in V2_KNOWN_POLLING_STATUSES:
                # Unknown status; keep polling until max wait elapses.
                pass

            if time.time() - start >= max_wait_s:
                raise ValueError("Timed out waiting for V2 H3 video generation.")

            sleep_poll(poll_interval_s)

        # ---- 3. Download and save the video --------------------------------
        video_path = make_output_path(prefix="minimax_h3_v2", ext=".mp4")
        download_file(video_url, video_path)

        video_obj: Any = None
        try:
            from comfy_api.latest import InputImpl
            video_obj = InputImpl.VideoFromFile(video_path)
        except ImportError:
            video_obj = video_path

        return (video_path, video_url, task_id, status, video_obj)
