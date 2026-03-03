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
        build_quad_masks_from_corners,
        build_overlay_images,
        image_batch_to_uint8_rgb_frames,
        normalize_image_batch,
        normalize_mask_batch,
    )
except ImportError:
    from overlay import (
        align_image_and_mask_batches,
        build_quad_masks_from_corners,
        build_overlay_images,
        image_batch_to_uint8_rgb_frames,
        normalize_image_batch,
        normalize_mask_batch,
    )

from flatsam.config import Config
from flatsam.flatsam import flatsam_track
from flatsam.utils.geom import H_warp
import flatsam.utils.geom as gu


def _coerce_point_list(value) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if arr.shape != (4, 2):
        return None
    return arr.T


def _parse_init_corners(value) -> Optional[np.ndarray]:
    """
    Parse init_corners into shape (2,4) float32.
    Invalid or displaced workflow values fall back to None.
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return _coerce_point_list(value)

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return _coerce_point_list(parsed)

    normalized = text.replace(";", ",").replace(" ", ",")
    parts = [part for part in normalized.split(",") if part.strip() != ""]
    if len(parts) != 8:
        return None

    try:
        vals = [float(part) for part in parts]
    except ValueError:
        return None

    arr = np.array([[vals[0], vals[1]],
                    [vals[2], vals[3]],
                    [vals[4], vals[5]],
                    [vals[6], vals[7]]], dtype=np.float32)
    return arr.T


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
                "overlay_source": (["tracked_quad", "input_mask"], {"default": "tracked_quad"}),
                "debug_tracking": ("BOOLEAN", {"default": False}),
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
        init_corners="",
        overlay_enable=True,
        overlay_opacity=0.35,
        overlay_mode="fill",
        overlay_color="red",
        debug_overlay=False,
        overlay_source="tracked_quad",
        debug_tracking=False,
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
                f"overlay_source={overlay_source}",
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
        if debug_tracking:
            print(
                "[WOFTSAM tracking] input batch:",
                f"T={int(track_images.shape[0])}",
                f"image_shape={tuple(track_images.shape)}",
                f"mask_shape={tuple(track_masks.shape)}",
                f"image_dtype={track_images.dtype}",
                f"mask_dtype={track_masks.dtype}",
            )
            if int(track_images.shape[0]) > 1:
                image_diff = float((track_images[1] - track_images[0]).abs().mean().item())
                mask0_sum = float(track_masks[0].sum().item())
                mask1_sum = float(track_masks[1].sum().item())
                mask_diff = float((track_masks[1] - track_masks[0]).abs().sum().item())
                print(
                    "[WOFTSAM tracking] frame deltas:",
                    f"image_mean_abs_diff_0_1={image_diff:.6f}",
                    f"mask_sum_0={mask0_sum:.2f}",
                    f"mask_sum_1={mask1_sum:.2f}",
                    f"mask_abs_diff_sum_0_1={mask_diff:.2f}",
                )

        frames = image_batch_to_uint8_rgb_frames(track_images)
        ext_masks = (track_masks > 0.5).to(dtype=torch.uint8).numpy()

        init_coords = _parse_init_corners(init_corners)
        if debug_overlay:
            print("[init_corners] raw=", repr(init_corners), "type=", type(init_corners))
            print("[init_corners] parsed=", None if init_coords is None else init_coords.tolist())
        if init_coords is None:
            init_coords = _init_coords_from_mask(ext_masks[0])

        # Minimal config, similar to demo_external_masks defaults
        conf = Config()
        conf.track_function = False
        if debug_tracking:
            try:
                node_hough_enabled = conf.hough_lines.enabled
            except Exception as exc:
                node_hough_enabled = f"<missing:{type(exc).__name__}>"
            print(
                "[WOFTSAM tracking] tracker config:",
                f"track_function={conf.track_function}",
                f"hough_lines.enabled={node_hough_enabled}",
                "demo_external_masks sets hough_lines.enabled=True, intersection_corners=False, resolve_symmetry.enabled=False, template_update=False",
            )

        # Name used in track_function call (seq_name in demo)
        seq_name = "comfyui_seq"

        track_function = conf.track_function if conf.track_function else flatsam_track
        if debug_tracking:
            print(
                "[WOFTSAM tracking] tracker call:",
                f"track_function={getattr(track_function, '__name__', type(track_function))}",
                f"frames_passed={len(frames)}",
                f"ext_masks_shape={ext_masks.shape}",
                f"ext_masks_dtype={ext_masks.dtype}",
            )

        all_corners = []
        h2init_history = []

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
            h2init_history.append(np.array(info.get("output_H2init", np.eye(3)), copy=True))
            if debug_tracking and frame_i in (0, 1):
                print(
                    "[WOFTSAM tracking] frame result:",
                    f"frame={frame_i}",
                    f"sam_mask_sum={float(np.asarray(sam_mask, dtype=np.float32).sum()):.2f}",
                    f"output_H2init={h2init_history[-1].tolist()}",
                    f"corners={all_corners[-1]}",
                )

        corners_json = json.dumps(all_corners, ensure_ascii=False)
        if debug_tracking:
            print(
                "[WOFTSAM tracking] corners structure:",
                f"type={type(all_corners)}",
                f"len={len(all_corners)}",
            )
            if all_corners:
                corners_np = np.asarray(all_corners, dtype=np.float32)
                ref = corners_np[0]
                max_delta = float(np.max(np.abs(corners_np - ref)))
                sample_indices = sorted({0, min(1, len(all_corners) - 1), min(10, len(all_corners) - 1), len(all_corners) - 1})
                for idx in sample_indices:
                    delta = float(np.max(np.abs(corners_np[idx] - ref)))
                    print(
                        "[WOFTSAM tracking] corners sample:",
                        f"frame={idx}",
                        f"corners={corners_np[idx].tolist()}",
                        f"max_abs_delta_from_frame0={delta:.6f}",
                    )
                print("[WOFTSAM tracking] corners max_delta_all_frames=", f"{max_delta:.6f}")
            if h2init_history:
                h2init_np = np.asarray(h2init_history, dtype=np.float32)
                h2init_max_delta = float(np.max(np.abs(h2init_np - h2init_np[0])))
                print("[WOFTSAM tracking] output_H2init max_delta_all_frames=", f"{h2init_max_delta:.6f}")
        if overlay_enable:
            if overlay_source == "tracked_quad":
                overlay_mask_batch = build_quad_masks_from_corners(
                    all_corners,
                    int(track_images.shape[1]),
                    int(track_images.shape[2]),
                    debug=debug_overlay,
                )
                if overlay_mask_batch is None:
                    print("[WOFTSAM overlay] warning: tracked quad mask generation failed, returning original images")
                    overlay_images = image_batch
                else:
                    overlay_images = build_overlay_images(
                        track_images,
                        overlay_mask_batch,
                        enabled=True,
                        opacity=overlay_opacity,
                        mode=overlay_mode,
                        color=overlay_color,
                        debug=debug_overlay,
                    )
            else:
                overlay_images = build_overlay_images(
                    track_images,
                    track_masks,
                    enabled=True,
                    opacity=overlay_opacity,
                    mode=overlay_mode,
                    color=overlay_color,
                    debug=debug_overlay,
                )
            if debug_overlay:
                delta_base = track_images if overlay_images.shape == track_images.shape else overlay_images
                overlay_delta = float((overlay_images - delta_base).abs().max().item()) if overlay_images.numel() > 0 else 0.0
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
