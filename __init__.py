"""MiniMax ComfyUI extension.

Re-exports the V1 (Hailuo) and V2 (H3) node classes plus their display-name
mappings so ComfyUI can discover both nodes when this directory is loaded as
a package under ``ComfyUI/custom_nodes``.
"""

from .minimax_video_generate import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
