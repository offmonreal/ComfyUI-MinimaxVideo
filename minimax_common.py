"""Shared helpers for MiniMax ComfyUI nodes.

Keeps authentication, HTTP transport, image conversion, polling utilities and
output-path handling in one place so V1 (Hailuo) and V2 (H3) node
implementations can stay focused on request construction and response parsing.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any, Dict, Optional

import folder_paths
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def _get_api_key(widget_key: str) -> str:
    """Resolve an API key using the documented priority order:

    1. Value typed directly into the node widget (highest priority).
    2. ``MINIMAX_API_KEY`` environment variable.
    3. ``config.json`` in the plugin root directory.

    The key is forwarded verbatim to the server; the server decides whether
    it is a Token Plan subscription key (``sk-cp-...``) or a normal PAYG key.
    """
    widget_key = (widget_key or "").strip()
    if widget_key:
        return widget_key

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


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def normalize_base_url(base_url: str, default_host: str) -> str:
    """Normalize a user-supplied base URL.

    The string must represent an API host (no version path). The caller is
    responsible for appending version-specific endpoint paths.
    """
    base_url = (base_url or "").strip()
    if not base_url:
        base_url = f"https://{default_host}"
    return base_url[:-1] if base_url.endswith("/") else base_url


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

def http_json_request(
    method: str,
    url: str,
    api_key: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Perform a JSON HTTP request and return the decoded response body.

    Raises ``ValueError`` with a descriptive English message on any HTTP
    error, including non-JSON error bodies.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body: Optional[bytes] = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        msg = raw.decode("utf-8", errors="replace").strip()
        raise ValueError(f"HTTP {e.code}: {msg}") from e
    return json.loads(raw.decode("utf-8"))


def download_file(url: str, out_path: str, timeout: int = 300) -> str:
    """Download a remote file to ``out_path`` and return the path."""
    req = urllib.request.Request(url=url, headers={"User-Agent": "ComfyUI"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


# ---------------------------------------------------------------------------
# Image conversion (ComfyUI IMAGE tensor -> data URL)
# ---------------------------------------------------------------------------

def tensor_to_pil_rgb(image_tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor to a PIL RGB image.

    Expects ``[B, H, W, C]`` with C = 3 or 4. Always returns RGB (drops alpha).
    """
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


def pil_to_data_url_jpeg(pil_img: Image.Image, quality: int = 95) -> str:
    """Encode a PIL image as a base64 JPEG data URL."""
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------

def make_output_path(prefix: str, ext: str) -> str:
    """Return a unique path under ComfyUI's output directory."""
    out_dir = folder_paths.get_output_directory()
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filename = f"{prefix}_{ts}_{int(time.time() * 1000) % 1000000:06d}{ext}"
    return os.path.join(out_dir, filename)


# ---------------------------------------------------------------------------
# Polling helpers
# ---------------------------------------------------------------------------

def sleep_poll(seconds: float) -> None:
    """Pause for ``seconds`` between polling attempts."""
    time.sleep(seconds)
