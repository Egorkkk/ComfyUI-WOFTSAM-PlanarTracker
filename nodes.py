import os
import sys
import json
from typing import List, Tuple, Optional

import numpy as np

# --- Make vendored flatsam importable ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_THIS_DIR, "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from flatsam.config import Config
from flatsam.flatsam import flatsam_track
from flatsam.utils.geom import H_warp
import flatsam.utils.geom as gu


def _comfy_image_to_uint8_rgb_batch(images: np.ndarray) -> List[np.ndarray]:
    """
    ComfyUI IMAGE is typically float32 in [0..1], shape [B,H,W,C] (C=3).
    Return list of uint8 RGB frames [H,W,3].
    """
    if images is None:
        raise ValueError("images is None")
    if not isinstance(images, np.ndarray):
        images = np.array(images)

    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError(f"Expected images shape [B,H,W,C], got {images.shape}")

    imgs = images[..., :3]
    imgs = np.clip(imgs, 0.0, 1.0)
    imgs_u8 = (imgs * 255.0).round().astype(np.uint8)
    return [imgs_u8[i] for i in range(imgs_u8.shape[0])]


def _comfy_mask_to_uint8_batch(masks: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    ComfyUI MASK is typically float32 in [0..1], shape [B,H,W] (sometimes [B,H,W,1]).
    Return np.uint8 array [B,H,W] with values 0/1.
    """
    if masks is None:
        raise ValueError("masks is None")
    if not isinstance(masks, np.ndarray):
        masks = np.array(masks)

    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]

    if masks.ndim != 3:
        raise ValueError(f"Expected masks shape [B,H,W] (or [B,H,W,1]), got {masks.shape}")

    m = (masks > threshold).astype(np.uint8)
    return m


def _parse_init_corners(s: str) -> Optional[np.ndarray]:
    """
    Parse "x1,y1,x2,y2,x3,y3,x4,y4" into init_coords shape (2,4) float32.
    Returns None if empty/blank.
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None

    # allow JSON too: [[x,y],...]
    if s.startswith("["):
        pts = json.loads(s)
        if not (isinstance(pts, list) and len(pts) == 4):
            raise ValueError("init_corners JSON must be a list of 4 [x,y] points")
        arr = np.array(pts, dtype=np.float32)  # (4,2)
        return arr.T  # (2,4)

    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip() != ""]
    if len(parts) != 8:
        raise ValueError("init_corners must contain 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4")

    vals = [float(p) for p in parts]
    arr = np.array([[vals[0], vals[1]],
                    [vals[2], vals[3]],
                    [vals[4], vals[5]],
                    [vals[6], vals[7]]], dtype=np.float32)  # (4,2)
    return arr.T  # (2,4)


def _init_coords_from_mask(mask0: np.ndarray) -> np.ndarray:
    bbox = gu.mask2bbox(mask0 > 0)  # same as demo_external_masks.py :contentReference[oaicite:6]{index=6}
    return np.asarray(bbox.as_points(), dtype=np.float32).T  # (2,4)


def _coords_2x4_to_list(coords_2x4: np.ndarray) -> List[List[float]]:
    # coords is (2,4) => list of 4 points [[x,y],...]
    xs = coords_2x4[0, :].tolist()
    ys = coords_2x4[1, :].tolist()
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


class WOFTSAM_Corners_Track:
    """
    ComfyUI node: track planar corners using WOFTSAM with externally provided per-frame masks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
            "optional": {
                # If empty: use bbox from first mask
                "init_corners": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("corners_json",)
    FUNCTION = "run"
    CATEGORY = "tracking/planar"

    def run(self, images, masks, init_corners=""):
        frames = _comfy_image_to_uint8_rgb_batch(images)
        ext_masks = _comfy_mask_to_uint8_batch(masks)  # (T,H,W) of 0/1

        # Align lengths defensively
        T = min(len(frames), int(ext_masks.shape[0]))
        if T == 0:
            raise ValueError("No frames or masks to process")
        if len(frames) != ext_masks.shape[0]:
            frames = frames[:T]
            ext_masks = ext_masks[:T]

        init_coords = _parse_init_corners(init_corners)
        if init_coords is None:
            init_coords = _init_coords_from_mask(ext_masks[0])

        # Minimal config, similar to demo_external_masks defaults
        conf = Config()
        conf.track_function = False

        # Name used in track_function call (seq_name in demo)
        seq_name = "comfyui_seq"

        track_function = conf.track_function if conf.track_function else flatsam_track

        all_corners = []

        # Call signature mirrors demo_external_masks.py :contentReference[oaicite:7]{index=7}
        for frame_i, sam_mask, _, info in track_function(
            None,
            conf,
            frames,
            init_coords,
            seq_name,
            external_masks=ext_masks,
        ):
            # Derive init->current homography as in demo :contentReference[oaicite:8]{index=8}
            if "output_H" in info:
                H_init2current = info["output_H"]
            else:
                H_init2current = np.linalg.inv(info["output_H2init"])

            current_corners = H_warp(H_init2current, init_coords)  # (2,4)
            all_corners.append(_coords_2x4_to_list(current_corners))

        corners_json = json.dumps(all_corners, ensure_ascii=False)
        return (corners_json,)


NODE_CLASS_MAPPINGS = {
    "WOFTSAM_Corners_Track": WOFTSAM_Corners_Track
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WOFTSAM_Corners_Track": "WOFTSAM Planar Corners (from masks)"
}