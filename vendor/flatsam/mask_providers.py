import logging
from timeit import default_timer as timer

import cv2
import einops
import numpy as np
import torch

import flatsam.utils.geom as gu
from flatsam.utils.io import ram_directory_of_images
from flatsam.utils.timing import general_time_measurer


logger = logging.getLogger(__name__)


class Sam2MaskProvider:
    def __init__(self, predictor, frames, init_mask, seq_name=None, estimate_on_init_frame=False, bbox_init=False):
        if predictor is None:
            raise ValueError("SAM2 mask provider requires a predictor.")

        self.frames = frames
        self._mask_iter = self._iter_masks(
            predictor,
            frames,
            init_mask,
            seq_name=seq_name,
            estimate_on_init_frame=estimate_on_init_frame,
            bbox_init=bbox_init,
        )

    def get_mask(self, frame_idx):
        next_frame_idx, mask = next(self._mask_iter)
        if next_frame_idx != frame_idx:
            raise ValueError(
                f"SAM2 mask provider expected sequential access. Requested frame {frame_idx}, got {next_frame_idx}."
            )
        return mask

    def _iter_masks(self, predictor, frames, init_mask, seq_name=None, estimate_on_init_frame=False, bbox_init=False):
        obj_id = 1
        start_time = timer()

        with ram_directory_of_images(frames, seq_name, double_first_frame=estimate_on_init_frame) as video_path:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(str(video_path), offload_video_to_cpu=True, async_loading_frames=True)

                if bbox_init:
                    init_bbox = gu.mask2bbox(init_mask).as_xyxy()
                    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=init_bbox)
                else:
                    predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=init_mask)

                duration_s = timer() - start_time
                logger.debug(f'SAM preparation: {duration_s}s')

                start_time = timer()
                n_timed_frames = 0
                last_frame_idx = -1
                sam_timer = general_time_measurer('sam_inner_tracking', cuda_sync=False, start_now=False)
                sam_timer.start()
                for current_frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
                    assert current_frame_idx == last_frame_idx + 1
                    last_frame_idx = current_frame_idx
                    n_timed_frames += 1
                    if estimate_on_init_frame and current_frame_idx == 0:
                        continue

                    out_mask = np.zeros(frames[0].shape[:2], dtype=bool)
                    for oid, mask_logit in zip(object_ids, mask_logits):
                        if oid == obj_id:
                            mask = (einops.rearrange(mask_logit, '1 H W -> H W') > 0).cpu().numpy()
                            out_mask = np.logical_or(out_mask, mask)
                    sam_timer.report(reduction='mean')

                    with torch.amp.autocast('cuda', enabled=False):
                        yield (current_frame_idx - 1 if estimate_on_init_frame else current_frame_idx, out_mask)
                    sam_timer.start()

                duration_s = float(timer() - start_time)
                ms_per_frame = (1000 * duration_s) / n_timed_frames
                fps = n_timed_frames / duration_s
                frame_h, frame_w = frames[0].shape[:2]
                logger.debug(f'SAM tracking on {frame_w}x{frame_h} images: {ms_per_frame:.0f} ms/frame, {fps:0.1f} FPS')


class ExternalMaskProvider:
    def __init__(self, frames, external_masks):
        self.frames = frames
        self.external_masks = self._normalize_collection(external_masks)
        if len(self.external_masks) < len(frames):
            raise ValueError(
                f"External masks length ({len(self.external_masks)}) is smaller than the number of frames ({len(frames)})."
            )

    def get_mask(self, frame_idx):
        if frame_idx >= len(self.frames):
            raise IndexError(f"Frame index {frame_idx} is out of range for {len(self.frames)} frames.")

        mask = np.asarray(self.external_masks[frame_idx])
        if mask.ndim != 2:
            raise ValueError(f"External mask at frame {frame_idx} must be 2D, got shape {mask.shape}.")

        frame_h, frame_w = self.frames[frame_idx].shape[:2]
        if mask.shape != (frame_h, frame_w):
            mask = cv2.resize(mask.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

        return mask > 0

    def _normalize_collection(self, external_masks):
        if isinstance(external_masks, np.ndarray):
            if external_masks.ndim != 3:
                raise ValueError(
                    f"External masks ndarray must have shape (T, H, W), got {external_masks.shape}."
                )
            return [external_masks[idx] for idx in range(external_masks.shape[0])]

        if isinstance(external_masks, list):
            return external_masks

        if isinstance(external_masks, tuple):
            return list(external_masks)

        raise ValueError("External masks must be a numpy array of shape (T, H, W) or a list of 2D masks.")


def build_mask_provider(
    predictor,
    conf,
    frames,
    init_mask,
    seq_name=None,
    estimate_on_init_frame=False,
    bbox_init=False,
    external_masks=None,
):
    mask_source = None
    if hasattr(conf, '__dict__'):
        mask_source = conf.__dict__.get('mask_source')

    use_external = external_masks is not None or mask_source == 'external'
    if use_external:
        if external_masks is None:
            raise ValueError("External mask source selected, but no external masks were provided.")
        return ExternalMaskProvider(frames, external_masks)

    return Sam2MaskProvider(
        predictor,
        frames,
        init_mask,
        seq_name=seq_name,
        estimate_on_init_frame=estimate_on_init_frame,
        bbox_init=bbox_init,
    )
