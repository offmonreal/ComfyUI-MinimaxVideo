"""MiniMax Hailuo Video (V1) ComfyUI node.

Targets the V1 ``video_generation`` API. Supports both normal PAYG API keys
and Token Plan subscription keys (``sk-cp-...``). The server decides
entitlement; the client forwards the key verbatim using Bearer auth.

All model / mode / resolution / duration combinations are validated locally
before any HTTP request is made so that the user receives a clear English
error rather than an opaque server-side rejection.
"""

import time
from typing import Any, Dict, Tuple

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

V1_DEFAULT_HOST = "api.minimax.io"
V1_PATH_VIDEO_GENERATION = "/v1/video_generation"
V1_PATH_QUERY = "/v1/query/video_generation"
V1_PATH_FILES_RETRIEVE = "/v1/files/retrieve"


# Models offered in the V1 dropdown.
V1_MODELS = [
    "MiniMax-Hailuo-2.3",
    "MiniMax-Hailuo-2.3-Fast",
    "MiniMax-Hailuo-02",
]


# Resolution / duration matrix per (model, mode). Resolution keys map to the
# list of allowed duration strings (kept as strings to match widget style).
V1_TEXT_TO_VIDEO_MATRIX: Dict[str, Dict[str, Any]] = {
    "MiniMax-Hailuo-2.3": {
        "resolutions": ["768P", "1080P"],
        "durations_by_resolution": {
            "768P": ["6", "10"],
            "1080P": ["6"],
        },
    },
    "MiniMax-Hailuo-02": {
        "resolutions": ["768P", "1080P"],
        "durations_by_resolution": {
            "768P": ["6", "10"],
            "1080P": ["6"],
        },
    },
}

V1_IMAGE_TO_VIDEO_MATRIX: Dict[str, Dict[str, Any]] = {
    "MiniMax-Hailuo-2.3": {
        "resolutions": ["768P", "1080P"],
        "durations_by_resolution": {
            "768P": ["6", "10"],
            "1080P": ["6"],
        },
    },
    "MiniMax-Hailuo-2.3-Fast": {
        "resolutions": ["768P", "1080P"],
        "durations_by_resolution": {
            "768P": ["6", "10"],
            "1080P": ["6"],
        },
    },
    "MiniMax-Hailuo-02": {
        "resolutions": ["512P", "768P", "1080P"],
        "durations_by_resolution": {
            "512P": ["6", "10"],
            "768P": ["6", "10"],
            "1080P": ["6"],
        },
    },
}

# First + Last Frame only supports Hailuo-02. 512P is explicitly disallowed
# by current MiniMax V1 documentation.
V1_FIRST_LAST_FRAME_RESOLUTIONS = ["768P", "1080P"]
V1_FIRST_LAST_FRAME_DURATIONS_BY_RESOLUTION: Dict[str, list] = {
    "768P": ["6", "10"],
    "1080P": ["6"],
}


# V1 statuses documented by MiniMax.
V1_STATUS_SUCCESS = "Success"
V1_STATUS_FAIL = "Fail"
V1_KNOWN_POLLING_STATUSES = {"Preparing", "Queueing", "Processing"}


# ---------------------------------------------------------------------------
# Local validation
# ---------------------------------------------------------------------------

def _validate_v1_text_to_video(model: str, resolution: str, duration: str) -> None:
    if model == "MiniMax-Hailuo-2.3-Fast":
        raise ValueError(
            "MiniMax-Hailuo-2.3-Fast does not support Text-to-Video. "
            "Use MiniMax-Hailuo-2.3 or MiniMax-Hailuo-02 instead."
        )
    matrix = V1_TEXT_TO_VIDEO_MATRIX.get(model)
    if matrix is None:
        raise ValueError(
            f"Model '{model}' does not support Text-to-Video in V1."
        )
    if resolution not in matrix["resolutions"]:
        raise ValueError(
            f"Resolution '{resolution}' is not allowed for {model} Text-to-Video. "
            f"Allowed: {', '.join(matrix['resolutions'])}."
        )
    allowed = matrix["durations_by_resolution"].get(resolution, [])
    if duration not in allowed:
        raise ValueError(
            f"Duration '{duration}s' is not allowed for {model} Text-to-Video "
            f"at {resolution}. Allowed: {', '.join(a + 's' for a in allowed)}."
        )


def _validate_v1_image_to_video(model: str, resolution: str, duration: str) -> None:
    matrix = V1_IMAGE_TO_VIDEO_MATRIX.get(model)
    if matrix is None:
        raise ValueError(
            f"Model '{model}' does not support Image-to-Video in V1."
        )
    if resolution not in matrix["resolutions"]:
        raise ValueError(
            f"Resolution '{resolution}' is not allowed for {model} Image-to-Video. "
            f"Allowed: {', '.join(matrix['resolutions'])}."
        )
    allowed = matrix["durations_by_resolution"].get(resolution, [])
    if duration not in allowed:
        raise ValueError(
            f"Duration '{duration}s' is not allowed for {model} Image-to-Video "
            f"at {resolution}. Allowed: {', '.join(a + 's' for a in allowed)}."
        )


def _validate_v1_first_last_frame(model: str, resolution: str, duration: str) -> None:
    if model != "MiniMax-Hailuo-02":
        raise ValueError(
            "First + Last Frame mode requires the MiniMax-Hailuo-02 model in V1. "
            f"Selected model: '{model}'."
        )
    if resolution == "512P":
        raise ValueError(
            "First + Last Frame mode does not support 512P resolution in V1."
        )
    if resolution not in V1_FIRST_LAST_FRAME_RESOLUTIONS:
        raise ValueError(
            f"Resolution '{resolution}' is not allowed for First + Last Frame in V1. "
            f"Allowed: {', '.join(V1_FIRST_LAST_FRAME_RESOLUTIONS)}."
        )
    allowed = V1_FIRST_LAST_FRAME_DURATIONS_BY_RESOLUTION.get(resolution, [])
    if duration not in allowed:
        raise ValueError(
            f"Duration '{duration}s' is not allowed for First + Last Frame at "
            f"{resolution}. Allowed: {', '.join(d + 's' for d in allowed)}."
        )


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class MinimaxVideoGenerate:
    """V1 Hailuo Video node.

    Preserves the historical ``MinimaxVideoGenerate`` class identifier so that
    workflows saved against this fork continue to resolve to a node class
    after the localization + V2 additions.
    """

    CATEGORY = "Ricksf-Toolbox"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("📁Video Path", "🔗Video URL", "🆔Task ID", "📌Status", "🎞️Video")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🎬Generation Mode": (["📝Text-to-Video", "🖼️Image-to-Video", "🧷First + Last Frame"],),
                "📝Prompt": ("STRING", {"multiline": True, "default": ""}),
                "🧠Model": (V1_MODELS, {"default": "MiniMax-Hailuo-2.3"}),
                "🖼️Resolution": (["768P", "1080P", "720P", "512P"], {"default": "768P"}),
                "⏱️Duration (seconds)": (["6", "10"], {"default": "6"}),
                "✨Prompt Optimizer": ("BOOLEAN", {"default": True}),
                "⚡Fast Preprocessing": ("BOOLEAN", {"default": False}),
                "🌐Base URL": ("STRING", {"default": f"https://{V1_DEFAULT_HOST}"}),
                "🔑API / Subscription Key": ("STRING", {"default": ""}),
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
        model = str(kwargs.get("🧠Model", "MiniMax-Hailuo-2.3"))
        resolution = str(kwargs.get("🖼️Resolution", "768P"))
        duration = int(kwargs.get("⏱️Duration (seconds)", "6"))
        prompt_optimizer = bool(kwargs.get("✨Prompt Optimizer", True))
        fast_preprocessing = bool(kwargs.get("⚡Fast Preprocessing", False))
        base_url_raw = str(kwargs.get("🌐Base URL", f"https://{V1_DEFAULT_HOST}"))
        api_key_widget = str(kwargs.get("🔑API / Subscription Key", "") or "")
        max_wait_s = int(kwargs.get("⌛Max Wait (seconds)", 1200))
        poll_interval_s = int(kwargs.get("🔁Poll Interval (seconds)", 10))
        ref_image = kwargs.get("🖼️Reference Image (for Image-to-Video)")
        first_frame_image = kwargs.get("🖼️First Frame Image")
        last_frame_image = kwargs.get("🖼️Last Frame Image")

        duration_str = str(duration)

        # ---- Authentication -------------------------------------------------
        api_key = _get_api_key(api_key_widget)
        if not api_key:
            raise ValueError(
                "No API key provided. Enter the API / Subscription Key on the node, "
                "set the MINIMAX_API_KEY environment variable, or configure it in config.json."
            )

        base_url = normalize_base_url(base_url_raw, V1_DEFAULT_HOST)

        # ---- Local validation ----------------------------------------------
        if mode == "📝Text-to-Video":
            _validate_v1_text_to_video(model, resolution, duration_str)
        elif mode == "🖼️Image-to-Video":
            _validate_v1_image_to_video(model, resolution, duration_str)
            if ref_image is None and first_frame_image is None:
                raise ValueError(
                    "Image-to-Video requires 🖼️Reference Image (for Image-to-Video) "
                    "or 🖼️First Frame Image."
                )
        elif mode == "🧷First + Last Frame":
            _validate_v1_first_last_frame(model, resolution, duration_str)
            if first_frame_image is None:
                raise ValueError("First + Last Frame requires 🖼️First Frame Image.")
            if last_frame_image is None:
                raise ValueError("First + Last Frame requires 🖼️Last Frame Image.")
        else:
            raise ValueError(f"Unknown generation mode: '{mode}'.")

        # ---- Build V1 request payload --------------------------------------
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "prompt_optimizer": prompt_optimizer,
            "fast_pretreatment": fast_preprocessing,
        }

        if mode == "🖼️Image-to-Video":
            chosen_ref = ref_image if ref_image is not None else first_frame_image
            ref_pil = tensor_to_pil_rgb(chosen_ref)
            payload["first_frame_image"] = pil_to_data_url_jpeg(ref_pil)

        elif mode == "🧷First + Last Frame":
            first_pil = tensor_to_pil_rgb(first_frame_image)
            last_pil = tensor_to_pil_rgb(last_frame_image)
            payload["first_frame_image"] = pil_to_data_url_jpeg(first_pil)
            payload["last_frame_image"] = pil_to_data_url_jpeg(last_pil)

        # ---- 1. Create the V1 task ----------------------------------------
        create_url = f"{base_url}{V1_PATH_VIDEO_GENERATION}"
        create_resp = http_json_request(
            method="POST",
            url=create_url,
            api_key=api_key,
            payload=payload,
        )
        task_id = str(create_resp.get("task_id") or "")
        if not task_id:
            raise ValueError(f"Failed to create V1 task, API returned: {create_resp}")

        # ---- 2. Poll V1 task status ----------------------------------------
        start = time.time()
        file_id = ""
        status = ""
        while True:
            get_url = f"{base_url}{V1_PATH_QUERY}?task_id={task_id}"
            task_resp = http_json_request(method="GET", url=get_url, api_key=api_key)
            status = str(task_resp.get("status") or "")

            if status == V1_STATUS_SUCCESS:
                file_id = str(task_resp.get("file_id") or "")
                break
            if status == V1_STATUS_FAIL:
                error_msg = task_resp.get("error_message", "Unknown error")
                raise ValueError(f"Video generation failed: {error_msg}")
            if status not in V1_KNOWN_POLLING_STATUSES:
                # Unknown status; keep polling until max wait elapses.
                pass

            if time.time() - start >= max_wait_s:
                raise ValueError("Timed out waiting for V1 video generation.")

            sleep_poll(poll_interval_s)

        if not file_id:
            raise ValueError("V1 task succeeded but no file_id was returned.")

        # ---- 3. Retrieve the file download URL ------------------------------
        file_url = f"{base_url}{V1_PATH_FILES_RETRIEVE}?file_id={file_id}"
        file_resp = http_json_request(method="GET", url=file_url, api_key=api_key)

        video_url = ""
        if "file" in file_resp and "download_url" in file_resp["file"]:
            video_url = file_resp["file"]["download_url"]
        else:
            raise ValueError(
                f"Unable to retrieve V1 file download URL, API returned: {file_resp}"
            )

        # ---- 4. Download and save the video --------------------------------
        video_path = make_output_path(prefix="minimax_hailuo_v1", ext=".mp4")
        download_file(video_url, video_path)

        video_obj: Any = None
        try:
            from comfy_api.latest import InputImpl
            video_obj = InputImpl.VideoFromFile(video_path)
        except ImportError:
            video_obj = video_path

        return (video_path, video_url, task_id, status, video_obj)
