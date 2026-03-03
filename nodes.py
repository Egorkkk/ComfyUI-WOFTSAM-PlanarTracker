import os
import sys
import json
from typing import List, Optional

import numpy as np
import torch

# --- Make vendored flatsam importable ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_THIS_DIR, "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

try:
    from .overlay import (
        align_image_and_mask_batches,
        build_overlay_images,
        image_batch_to_uint8_rgb_frames,
        normalize_image_batch,
        normalize_mask_batch,
    )
except ImportError:
    from overlay import (
        align_image_and_mask_batches,
        build_overlay_images,
        image_batch_to_uint8_rgb_frames,
        normalize_image_batch,
        normalize_mask_batch,
    )

from flatsam.config import Config
from flatsam.flatsam import flatsam_track
from flatsam.utils.geom import H_warp
import flatsam.utils.geom as gu


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


def _tensor_min_max(value):
    if value.numel() == 0:
        return (0.0, 0.0)
    return (float(value.min().item()), float(value.max().item()))


def _coerce_overlay_enable(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "":
            return True
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        return True
    if value is None:
        return True
    return bool(value)


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
                "overlay_enable": ("BOOLEAN", {"default": True}),
                "overlay_opacity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "overlay_mode": (["fill", "outline"], {"default": "fill"}),
                "overlay_color": (["red", "green", "blue"], {"default": "red"}),
                "debug_overlay": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("corners_json", "overlay_images")
    FUNCTION = "run"
    CATEGORY = "tracking/planar"

    def run(
        self,
        images,
        masks,
        overlay_enable=True,
        overlay_opacity=0.35,
        overlay_mode="fill",
        overlay_color="red",
        debug_overlay=False,
        init_corners="",
    ):
        overlay_enable_raw = overlay_enable
        overlay_enable = _coerce_overlay_enable(overlay_enable)
        if debug_overlay:
            raw_mask_tensor = masks.detach() if torch.is_tensor(masks) else torch.as_tensor(np.asarray(masks))
            raw_mask_min, raw_mask_max = _tensor_min_max(raw_mask_tensor.to(torch.float32))
            print(
                "[WOFTSAM overlay] input:",
                f"images_shape={tuple(images.shape) if hasattr(images, 'shape') else type(images)}",
                f"images_dtype={getattr(images, 'dtype', type(images))}",
                f"masks_shape={tuple(raw_mask_tensor.shape)}",
                f"masks_dtype={raw_mask_tensor.dtype}",
                f"masks_min={raw_mask_min:.4f}",
                f"masks_max={raw_mask_max:.4f}",
                f"overlay_enable_raw={overlay_enable_raw!r}",
                f"overlay_enable={overlay_enable}",
                f"overlay_opacity={overlay_opacity}",
                f"overlay_mode={overlay_mode}",
            )

        image_batch = normalize_image_batch(images)
        mask_batch = normalize_mask_batch(masks)
        if debug_overlay:
            norm_mask_min, norm_mask_max = _tensor_min_max(mask_batch)
            print(
                "[WOFTSAM overlay] normalized:",
                f"images_shape={tuple(image_batch.shape)}",
                f"images_dtype={image_batch.dtype}",
                f"masks_shape={tuple(mask_batch.shape)}",
                f"masks_dtype={mask_batch.dtype}",
                f"masks_min={norm_mask_min:.4f}",
                f"masks_max={norm_mask_max:.4f}",
            )
        track_images, track_masks = align_image_and_mask_batches(image_batch, mask_batch)
        if debug_overlay:
            aligned_mask_min, aligned_mask_max = _tensor_min_max(track_masks)
            print(
                "[WOFTSAM overlay] aligned:",
                f"images_shape={tuple(track_images.shape)}",
                f"masks_shape={tuple(track_masks.shape)}",
                f"masks_min={aligned_mask_min:.4f}",
                f"masks_max={aligned_mask_max:.4f}",
            )

        frames = image_batch_to_uint8_rgb_frames(track_images)
        ext_masks = (track_masks > 0.5).to(dtype=torch.uint8).numpy()

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
        overlay_masks = []

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
            overlay_masks.append(torch.as_tensor(np.asarray(sam_mask), dtype=torch.float32))

        corners_json = json.dumps(all_corners, ensure_ascii=False)
        if overlay_enable:
            overlay_mask_batch = torch.stack(overlay_masks, dim=0) if overlay_masks else track_masks
            overlay_images = build_overlay_images(
                track_images,
                overlay_mask_batch,
                enabled=True,
                opacity=overlay_opacity,
                mode=overlay_mode,
                color=overlay_color,
                debug=debug_overlay,
            )
            if debug_overlay:
                overlay_delta = float((overlay_images - track_images).abs().max().item()) if overlay_images.numel() > 0 else 0.0
                print(
                    "[WOFTSAM overlay] result:",
                    f"overlay_shape={tuple(overlay_images.shape)}",
                    f"overlay_delta_max={overlay_delta:.4f}",
                )
        else:
            if debug_overlay:
                print("[WOFTSAM overlay] branch: overlay disabled, returning original images")
            overlay_images = image_batch

        return (corners_json, overlay_images)


NODE_CLASS_MAPPINGS = {
    "WOFTSAM_Corners_Track": WOFTSAM_Corners_Track
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WOFTSAM_Corners_Track": "WOFTSAM Planar Corners (from masks)"
}
