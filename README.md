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
