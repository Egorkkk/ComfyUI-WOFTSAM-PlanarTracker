import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)

_OVERLAY_COLORS = {
    "red": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    "green": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
    "blue": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
}


def normalize_image_batch(images):
    if images is None:
        raise ValueError("images is None")

    if torch.is_tensor(images):
        image_batch = images.detach().cpu().to(torch.float32)
    else:
        image_batch = torch.as_tensor(np.asarray(images), dtype=torch.float32)

    if image_batch.ndim != 4 or image_batch.shape[-1] not in (3, 4):
        raise ValueError(f"Expected images shape [B,H,W,C], got {tuple(image_batch.shape)}")

    return image_batch[..., :3].clamp(0.0, 1.0).contiguous()


def normalize_mask_batch(masks):
    if masks is None:
        raise ValueError("masks is None")

    if torch.is_tensor(masks):
        mask_batch = masks.detach().cpu().to(torch.float32)
    else:
        mask_batch = torch.as_tensor(np.asarray(masks), dtype=torch.float32)

    if mask_batch.ndim == 4:
        if mask_batch.shape[1] == 1:
            mask_batch = mask_batch[:, 0]
        elif mask_batch.shape[-1] == 1:
            mask_batch = mask_batch[..., 0]
        else:
            raise ValueError(
                f"Expected masks shape [B,H,W], [B,1,H,W], or [B,H,W,1], got {tuple(mask_batch.shape)}"
            )

    if mask_batch.ndim != 3:
        raise ValueError(
            f"Expected masks shape [B,H,W], [B,1,H,W], or [B,H,W,1], got {tuple(mask_batch.shape)}"
        )

    if mask_batch.numel() > 0 and float(mask_batch.max().item()) > 1.0:
        mask_batch = mask_batch / 255.0

    return mask_batch.clamp(0.0, 1.0).contiguous()


def align_image_and_mask_batches(image_batch, mask_batch):
    if image_batch.shape[0] == 0 or mask_batch.shape[0] == 0:
        raise ValueError("No frames or masks to process")

    target_batch = min(int(image_batch.shape[0]), int(mask_batch.shape[0]))
    if image_batch.shape[0] != mask_batch.shape[0]:
        logger.warning(
            "WOFTSAM overlay: image batch (%d) and mask batch (%d) differ, cropping to %d frames.",
            int(image_batch.shape[0]),
            int(mask_batch.shape[0]),
            target_batch,
        )

    image_batch = image_batch[:target_batch]
    mask_batch = mask_batch[:target_batch]

    target_hw = tuple(int(v) for v in image_batch.shape[1:3])
    if tuple(int(v) for v in mask_batch.shape[1:3]) != target_hw:
        logger.warning(
            "WOFTSAM overlay: resizing masks from %s to %s with nearest-neighbor for preview/tracking alignment.",
            tuple(int(v) for v in mask_batch.shape[1:3]),
            target_hw,
        )
        mask_batch = F.interpolate(mask_batch.unsqueeze(1), size=target_hw, mode="nearest").squeeze(1)

    return image_batch.contiguous(), mask_batch.contiguous()


def image_batch_to_uint8_rgb_frames(image_batch):
    image_uint8 = (image_batch.numpy() * 255.0).round().astype(np.uint8)
    return [image_uint8[i] for i in range(image_uint8.shape[0])]


def _mask_outline(mask_batch):
    mask = (mask_batch > 0.5).to(torch.float32).unsqueeze(1)
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    outline = (dilated > 0.5) & (eroded < 0.5)
    return outline.squeeze(1)


def build_quad_masks_from_corners(corners_list, height, width, debug=False):
    if not corners_list:
        return None

    quad_masks = []
    for frame_idx, corners in enumerate(corners_list):
        try:
            quad = np.asarray(corners, dtype=np.float32)
        except (TypeError, ValueError):
            print(f"[WOFTSAM overlay] warning: invalid tracked quad at frame {frame_idx}, falling back to original images")
            return None

        if quad.shape != (4, 2):
            print(f"[WOFTSAM overlay] warning: expected tracked quad shape (4, 2) at frame {frame_idx}, got {quad.shape}")
            return None

        quad_int = np.rint(quad).astype(np.int32)
        quad_int[:, 0] = np.clip(quad_int[:, 0], 0, width - 1)
        quad_int[:, 1] = np.clip(quad_int[:, 1], 0, height - 1)

        mask_quad = np.zeros((height, width), dtype=np.float32)
        cv2.fillConvexPoly(mask_quad, quad_int, 1.0)
        quad_masks.append(mask_quad)

        if debug and frame_idx < 2:
            print(
                "[WOFTSAM overlay] tracked_quad:",
                f"frame={frame_idx}",
                f"quad={quad_int.tolist()}",
                f"mask_min={float(mask_quad.min()):.4f}",
                f"mask_max={float(mask_quad.max()):.4f}",
                f"mask_sum={float(mask_quad.sum()):.1f}",
            )

    return torch.from_numpy(np.stack(quad_masks, axis=0))


def build_overlay_images(image_batch, mask_batch, enabled=True, opacity=0.35, mode="fill", color="red", debug=False):
    if not enabled:
        if debug:
            print("[WOFTSAM overlay] build skipped: enabled=False")
        return image_batch

    image_batch, mask_batch = align_image_and_mask_batches(image_batch, mask_batch)
    alpha = max(0.0, min(1.0, float(opacity)))
    if alpha == 0.0:
        if debug:
            print("[WOFTSAM overlay] build skipped: opacity=0")
        return image_batch

    if mode == "fill":
        alpha_mask = mask_batch.to(image_batch.dtype).unsqueeze(-1) * alpha
    elif mode == "outline":
        alpha_mask = _mask_outline(mask_batch).to(image_batch.dtype).unsqueeze(-1) * alpha
    else:
        raise ValueError(f"Unsupported overlay_mode: {mode}")

    color_value = _OVERLAY_COLORS.get(color, _OVERLAY_COLORS["red"]).to(image_batch.device)
    blended = image_batch * (1.0 - alpha_mask) + color_value.view(1, 1, 1, 3) * alpha_mask
    if debug:
        print(
            "[WOFTSAM overlay] build:",
            f"mode={mode}",
            f"alpha={alpha}",
            f"mask_min={float(mask_batch.min().item()):.4f}",
            f"mask_max={float(mask_batch.max().item()):.4f}",
            f"alpha_mask_max={float(alpha_mask.max().item()):.4f}",
            f"image_shape={tuple(image_batch.shape)}",
        )
    return blended.clamp(0.0, 1.0).contiguous()
