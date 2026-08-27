import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import folder_paths


@dataclass(frozen=True)
class _MinimaxClientConfig:
    base_url: str
    api_key: str


def _normalize_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        return "https://api.minimaxi.com/v1"
    return base_url[:-1] if base_url.endswith("/") else base_url


def _get_api_key(api_key: str) -> str:
    # Priority: Node widget input > Environment variable > Local config
    api_key = (api_key or "").strip()
    if api_key:
        return api_key

    env_key = (os.environ.get("MINIMAX_API_KEY") or "").strip()
    if env_key:
        return env_key

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                config_key = (cfg.get("MINIMAX_API_KEY") or "").strip()
                if config_key:
                    return config_key
        except Exception:
            pass

    return ""


def _http_json_request(method: str, url: str, api_key: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body: Optional[bytes] = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        msg = raw.decode("utf-8", errors="replace").strip()
        raise ValueError(f"HTTP {e.code}: {msg}") from e
    return json.loads(raw.decode("utf-8"))


def _download_file(url: str, out_path: str) -> str:
    req = urllib.request.Request(url=url, headers={"User-Agent": "ComfyUI"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def _tensor_to_pil_rgb(image_tensor) -> Image.Image:
    if image_tensor is None:
        raise ValueError("image is empty")
    if len(image_tensor.shape) != 4 or image_tensor.shape[-1] not in (3, 4):
        raise ValueError("image tensor shape must be [B,H,W,C] with C being 3 or 4")
    img = image_tensor[0].detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).round().astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return Image.fromarray(img, mode="RGB")


def _pil_to_data_url_jpeg(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _make_output_path(prefix: str, ext: str) -> str:
    out_dir = folder_paths.get_output_directory()
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filename = f"{prefix}_{ts}_{int(time.time() * 1000) % 1000000:06d}{ext}"
    return os.path.join(out_dir, filename)


class MinimaxVideoGenerate:
    CATEGORY = "Ricksf-Toolbox"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("📁Video Path", "🔗Video URL", "🆔Task ID", "📌Status", "🎞️Video")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🎬Generation Mode": (["📝Text-to-Video", "🖼️Image-to-Video", "🧷First-and-Last-Frame Video"],),
                "📝Prompt": ("STRING", {"multiline": True, "default": ""}),
                "🧠Model ID": (["MiniMax-Hailuo-2.3", "MiniMax-Hailuo-02", "T2V-01-Director", "T2V-01", "S2V-01"], {"default": "MiniMax-Hailuo-2.3"}),
                "🖼️Resolution": (["768P", "1080P", "720P"], {"default": "768P"}),
                "⏱️Duration (seconds)": (["6", "10"], {"default": "6"}),
                "✨Auto-Optimize Prompt": ("BOOLEAN", {"default": True}),
                "⚡Fast Pretreatment": ("BOOLEAN", {"default": False}),
                "💧Add Watermark": ("BOOLEAN", {"default": False}),
                "🌐Base URL": ("STRING", {"default": "https://api.minimaxi.com/v1"}),
                "🔑API Key": ("STRING", {"default": ""}),
                "⌛Max Wait (seconds)": ("INT", {"default": 1200, "min": 10, "max": 3600}),
                "🔁Poll Interval (seconds)": ("INT", {"default": 10, "min": 5, "max": 60}),
            },
            "optional": {
                "🖼️Reference Image (for Image-to-Video)": ("IMAGE",),
                "🖼️First Frame Image": ("IMAGE",),
                "🖼️Last Frame Image": ("IMAGE",),
            },
        }

    def generate(self, **kwargs) -> Tuple[str, str, str, str, Any]:
        mode = str(kwargs.get("🎬Generation Mode", "")).strip()
        prompt = str(kwargs.get("📝Prompt", "") or "")
        model = str(kwargs.get("🧠Model ID", "MiniMax-Hailuo-2.3"))
        resolution = str(kwargs.get("🖼️Resolution", "768P"))
        duration = int(kwargs.get("⏱️Duration (seconds)", "6"))
        prompt_optimizer = bool(kwargs.get("✨Auto-Optimize Prompt", True))
        fast_pretreatment = bool(kwargs.get("⚡Fast Pretreatment", False))
        aigc_watermark = bool(kwargs.get("💧Add Watermark", False))
        base_url = str(kwargs.get("🌐Base URL", "https://api.minimaxi.com/v1"))
        api_key = str(kwargs.get("🔑API Key", "") or "")
        max_wait_s = int(kwargs.get("⌛Max Wait (seconds)", 1200))
        poll_interval_s = int(kwargs.get("🔁Poll Interval (seconds)", 10))
        ref_image = kwargs.get("🖼️Reference Image (for Image-to-Video)")
        first_frame_image = kwargs.get("🖼️First Frame Image")
        last_frame_image = kwargs.get("🖼️Last Frame Image")

        api_key_final = _get_api_key(api_key)
        if not api_key_final:
            raise ValueError(
                "No API Key provided. Please enter the API Key on the node, "
                "set the MINIMAX_API_KEY environment variable, or configure it in config.json."
            )

        cfg = _MinimaxClientConfig(base_url=_normalize_base_url(base_url), api_key=api_key_final)

        payload = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "prompt_optimizer": prompt_optimizer,
            "fast_pretreatment": fast_pretreatment,
            "aigc_watermark": aigc_watermark,
        }

        if mode == "🖼️Image-to-Video":
            chosen_ref = ref_image if ref_image is not None else first_frame_image
            if chosen_ref is None:
                raise ValueError(
                    "Image-to-Video requires 🖼️Reference Image (for Image-to-Video) "
                    "or 🖼️First Frame Image"
                )
            ref_pil = _tensor_to_pil_rgb(chosen_ref)
            payload["first_frame_image"] = _pil_to_data_url_jpeg(ref_pil)

        elif mode == "🧷First-and-Last-Frame Video":
            if first_frame_image is None:
                raise ValueError("First-and-Last-Frame Video requires 🖼️First Frame Image")
            if last_frame_image is None:
                raise ValueError("First-and-Last-Frame Video requires 🖼️Last Frame Image")
            first_pil = _tensor_to_pil_rgb(first_frame_image)
            last_pil = _tensor_to_pil_rgb(last_frame_image)
            payload["first_frame_image"] = _pil_to_data_url_jpeg(first_pil)
            payload["last_frame_image"] = _pil_to_data_url_jpeg(last_pil)

            if payload["model"] != "MiniMax-Hailuo-02":
                payload["model"] = "MiniMax-Hailuo-02"

        # 1. Create the task
        create_url = f"{cfg.base_url}/video_generation"
        create_resp = _http_json_request(
            method="POST",
            url=create_url,
            api_key=cfg.api_key,
            payload=payload,
        )
        task_id = str(create_resp.get("task_id") or "")
        if not task_id:
            raise ValueError(f"Failed to create task, API returned: {create_resp}")

        # 2. Poll task status
        start = time.time()
        file_id = ""
        status = ""
        while True:
            get_url = f"{cfg.base_url}/query/video_generation?task_id={task_id}"
            task_resp = _http_json_request(method="GET", url=get_url, api_key=cfg.api_key)
            status = str(task_resp.get("status") or "")

            if status == "Success":
                file_id = str(task_resp.get("file_id") or "")
                break
            elif status == "Fail":
                error_msg = task_resp.get("error_message", "Unknown error")
                raise ValueError(f"Video generation failed: {error_msg}")
            elif status not in ["Queueing", "Processing"]:
                # Unknown completion status; continue polling.
                pass

            if time.time() - start >= max_wait_s:
                raise ValueError("Timed out waiting for video generation")

            time.sleep(poll_interval_s)

        if not file_id:
            raise ValueError("Task succeeded but no file_id was returned")

        # 3. Retrieve the file download URL
        file_retrieve_url = f"{cfg.base_url}/files/retrieve?file_id={file_id}"
        file_resp = _http_json_request(method="GET", url=file_retrieve_url, api_key=cfg.api_key)

        video_url = ""
        if "file" in file_resp and "download_url" in file_resp["file"]:
            video_url = file_resp["file"]["download_url"]
        else:
            raise ValueError(f"Unable to retrieve file download URL, API returned: {file_resp}")

        # 4. Download and save the video
        video_path = _make_output_path(prefix="minimax_video", ext=".mp4")
        _download_file(video_url, video_path)

        video_obj: Any = None
        try:
            from comfy_api.latest import InputImpl
            video_obj = InputImpl.VideoFromFile(video_path)
        except ImportError:
            video_obj = video_path

        return (video_path, video_url, task_id, status, video_obj)


NODE_CLASS_MAPPINGS = {
    "MinimaxVideoGenerate": MinimaxVideoGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxVideoGenerate": "🤖Minimax Video Generate",
}
