# ComfyUI-WOFTSAM-PlanarTracker (Corners)

A ComfyUI custom node that runs WOFTSAM planar tracking using externally provided per-frame masks (e.g., from SAM3 nodes) and outputs tracked planar corners as JSON.

## Install
1) Copy this folder into `ComfyUI/custom_nodes/`.
2) Install requirements in your ComfyUI python environment:
   - `pip install -r custom_nodes/ComfyUI-WOFTSAM-PlanarTracker/requirements.txt`

Restart ComfyUI.

## Node
**WOFTSAM Planar Corners (from masks)**

Inputs:
- `images` (IMAGE batch)
- `masks` (MASK batch) - must be per-frame masks aligned with the images
- `init_corners` (optional STRING):
  - empty => corners are inferred from bbox of the first mask
  - can be "x1,y1,x2,y2,x3,y3,x4,y4" or JSON `[[x,y],[x,y],[x,y],[x,y]]`

Output:
- `corners_json` (STRING): list of frames, each frame contains 4 corner points [[x,y],...]
- `overlay_images` (IMAGE batch): preview frames with mask overlay for quick validation in `Preview Image` or video save nodes

Overlay controls:
- `overlay_enable`: when false, `overlay_images` returns the input `images` unchanged
- `overlay_opacity`: alpha for the mask preview
- `overlay_mode`: `fill` (alpha-blend) or `outline` (torch-generated contour)
- `overlay_color`: `red`, `green`, or `blue`

Reference behavior:
- Upstream `WOFTSAM/demo_external_masks.py` renders preview via `vu.blend_mask(..., alpha=0.2)`, which is a translucent mask overlay with contour. This node mirrors that preview intent with a portable torch implementation.

Quick use:
- Connect `overlay_images` to `Preview Image` to inspect the tracked mask overlay per frame.
- Or send `overlay_images` into VideoHelperSuite to write a preview video while keeping `corners_json` for downstream tracking data.
