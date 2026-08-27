"""Entry-point module for the MiniMax ComfyUI extension.

Re-exports the V1 (Hailuo) and V2 (H3) node classes plus their display-name
mappings so ComfyUI can discover both nodes through this module.
"""

from .minimax_hailuo_v1 import MinimaxVideoGenerate
from .minimax_h3_v2 import MinimaxH3VideoGenerate


NODE_CLASS_MAPPINGS = {
    "MinimaxVideoGenerate": MinimaxVideoGenerate,
    "MinimaxH3VideoGenerate": MinimaxH3VideoGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxVideoGenerate": "MiniMax Hailuo Video (V1)",
    "MinimaxH3VideoGenerate": "MiniMax H3 Video (V2)",
}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MinimaxVideoGenerate",
    "MinimaxH3VideoGenerate",
]
