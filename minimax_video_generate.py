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
    # 优先级: 节点 Widget 输入 > 环境变量 > 本地 Config
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
        raise ValueError("image 为空")
    if len(image_tensor.shape) != 4 or image_tensor.shape[-1] not in (3, 4):
        raise ValueError("image 张量形状必须是 [B,H,W,C] 且 C 为 3 或 4")
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
    RETURN_NAMES = ("📁视频路径", "🔗视频链接", "🆔任务ID", "📌状态", "🎞️视频")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "🎬生成模式": (["📝文生视频", "🖼️图生视频", "🧷首尾帧视频"],),
                "📝提示词": ("STRING", {"multiline": True, "default": ""}),
                "🧠模型ID": (["MiniMax-Hailuo-2.3", "MiniMax-Hailuo-02", "T2V-01-Director", "T2V-01", "S2V-01"], {"default": "MiniMax-Hailuo-2.3"}),
                "🖼️分辨率": (["768P", "1080P", "720P"], {"default": "768P"}),
                "⏱️时长(秒)": (["6", "10"], {"default": "6"}),
                "✨自动优化Prompt": ("BOOLEAN", {"default": True}),
                "⚡快速预处理": ("BOOLEAN", {"default": False}),
                "💧添加水印": ("BOOLEAN", {"default": False}),
                "🌐BaseURL": ("STRING", {"default": "https://api.minimaxi.com/v1"}),
                "🔑API Key": ("STRING", {"default": ""}),
                "⌛最大等待(秒)": ("INT", {"default": 1200, "min": 10, "max": 3600}),
                "🔁轮询间隔(秒)": ("INT", {"default": 10, "min": 5, "max": 60}),
            },
            "optional": {
                "🖼️参考图(用于图生)": ("IMAGE",),
                "🖼️首帧图": ("IMAGE",),
                "🖼️尾帧图": ("IMAGE",),
            },
        }

    def generate(self, **kwargs) -> Tuple[str, str, str, str, Any]:
        mode = str(kwargs.get("🎬生成模式", "")).strip()
        prompt = str(kwargs.get("📝提示词", "") or "")
        model = str(kwargs.get("🧠模型ID", "MiniMax-Hailuo-2.3"))
        resolution = str(kwargs.get("🖼️分辨率", "768P"))
        duration = int(kwargs.get("⏱️时长(秒)", "6"))
        prompt_optimizer = bool(kwargs.get("✨自动优化Prompt", True))
        fast_pretreatment = bool(kwargs.get("⚡快速预处理", False))
        aigc_watermark = bool(kwargs.get("💧添加水印", False))
        base_url = str(kwargs.get("🌐BaseURL", "https://api.minimaxi.com/v1"))
        api_key = str(kwargs.get("🔑API Key", "") or "")
        max_wait_s = int(kwargs.get("⌛最大等待(秒)", 1200))
        poll_interval_s = int(kwargs.get("🔁轮询间隔(秒)", 10))
        ref_image = kwargs.get("🖼️参考图(用于图生)")
        first_frame_image = kwargs.get("🖼️首帧图")
        last_frame_image = kwargs.get("🖼️尾帧图")

        api_key_final = _get_api_key(api_key)
        if not api_key_final:
            raise ValueError("未提供 API Key，请在节点输入 API Key，或设置环境变量 MINIMAX_API_KEY，或在 config.json 中配置。")

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

        if mode == "🖼️图生视频":
            chosen_ref = ref_image if ref_image is not None else first_frame_image
            if chosen_ref is None:
                raise ValueError("图生视频需要提供 🖼️参考图(用于图生) 或 🖼️首帧图")
            ref_pil = _tensor_to_pil_rgb(chosen_ref)
            payload["first_frame_image"] = _pil_to_data_url_jpeg(ref_pil)
            
        elif mode == "🧷首尾帧视频":
            if first_frame_image is None:
                raise ValueError("首尾帧视频需要提供 🖼️首帧图")
            if last_frame_image is None:
                raise ValueError("首尾帧视频需要提供 🖼️尾帧图")
            first_pil = _tensor_to_pil_rgb(first_frame_image)
            last_pil = _tensor_to_pil_rgb(last_frame_image)
            payload["first_frame_image"] = _pil_to_data_url_jpeg(first_pil)
            payload["last_frame_image"] = _pil_to_data_url_jpeg(last_pil)
            
            if payload["model"] != "MiniMax-Hailuo-02":
                payload["model"] = "MiniMax-Hailuo-02"

        # 1. 创建任务
        create_url = f"{cfg.base_url}/video_generation"
        create_resp = _http_json_request(
            method="POST",
            url=create_url,
            api_key=cfg.api_key,
            payload=payload,
        )
        task_id = str(create_resp.get("task_id") or "")
        if not task_id:
            raise ValueError(f"创建任务失败，API 返回：{create_resp}")

        # 2. 轮询任务状态
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
                error_msg = task_resp.get("error_message", "未知错误")
                raise ValueError(f"视频生成失败: {error_msg}")
            elif status not in ["Queueing", "Processing"]:
                # 如果有其他未知的完成状态
                pass
            
            if time.time() - start >= max_wait_s:
                raise ValueError("等待视频生成超时")
                
            time.sleep(poll_interval_s)

        if not file_id:
            raise ValueError("任务成功但未返回 file_id")

        # 3. 获取文件下载链接
        file_retrieve_url = f"{cfg.base_url}/files/retrieve?file_id={file_id}"
        file_resp = _http_json_request(method="GET", url=file_retrieve_url, api_key=cfg.api_key)
        
        video_url = ""
        if "file" in file_resp and "download_url" in file_resp["file"]:
            video_url = file_resp["file"]["download_url"]
        else:
            raise ValueError(f"无法获取文件下载链接，API 返回：{file_resp}")

        # 4. 下载视频并保存
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
    "MinimaxVideoGenerate": "🤖Minimax视频生成",
}
