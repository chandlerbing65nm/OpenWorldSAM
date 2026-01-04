## Demo Inference Scripts

The `demo/` directory provides four scripts that run OpenWorldSAM2 in inference mode for different segmentation tasks. Each script expects a Detectron2 config file, a checkpoint, an input image, and a list of prompts supplied on the command line.

### Notes

- Instance/semantic/panoptic scripts require prompts that **match the dataset taxonomy** so that prompts can be mapped to contiguous IDs. If a prompt is not found, the script raises a descriptive error. **TODO:** support prompts outside the dataset taxonomy. This requires the user to define a demo `MetadataCatalog` with class names, class colors for visualization, and class IDs. (something similar to: https://github.com/microsoft/X-Decoder/blob/main/inference_demo/demo_instseg.py).
- Referring segmentation uses dummy class IDs, so prompts may be any free-form expressions, but ensure that the selected config/weights were trained for referring tasks.
- All visualizations rely on `utils.visualizer.SegmentationResultVisualizer`, and outputs will be written to the path specified via `--output` (directories are created automatically).

### Common requirements

- Download/prepare the dataset metadata referenced in the config (`cfg.DATASETS.TEST[0]`) so that prompts can be mapped to the correct contiguous class IDs.
- Obtain model weights compatible with the chosen config.

All scripts share the following arguments:

| Argument | Description |
| --- | --- |
| `--config-file` | Path to a Detectron2 YAML config. |
| `--weights` | Path to the trained OpenWorldSAM2 checkpoint (`MODEL.WEIGHTS`). |
| `--image` | Path to the input image (RGB/BGR as expected by the config). |
| `--prompts ...` | Space-separated list of textual prompts. Instance/semantic/panoptic prompts must match dataset class names; referring prompts can be free-form expressions. |
| `--output` | Path where the visualization PNG will be saved. |
| `--opts ...` | Optional config overrides appended to the Detectron2 configuration list. |

Run each script from the repository root:

```bash
python demo/<script>.py --config-file <cfg.yaml> --weights <model.pth> --image /path/to/img.jpg --prompts "<prompt 1>" "<prompt 2>" ...
```

### Instance segmentation (`instance_inference.py`)

Produces per-instance masks, scores, and boxes for the requested categories and saves a visualization via Detectron2’s `Visualizer`. On COCO for reference:

```bash
python demo/instance_inference.py \
  --config-file configs/coco/instance-segmentation/Open-World-SAM2-CrossAttention.yaml \
  --weights "checkpoints/model_final.pth" \
  --image demo/images/giraffe.jpg \
  --prompts "giraffe" \
  --output demo/outputs/giraffe_instance.png
```

Adjusting post-processing thresholds for instance segmentation:

```python
# NMS threshold, determines how much overlap is allowed between bounding boxes for the same object in computer vision; a higher threshold (e.g., 0.7) is more permissive, keeping more boxes but risking duplicates, while a lower threshold (e.g., 0.3) is stricter, removing more boxes.
cfg.MODEL.OpenWorldSAM2.TEST.NMS_THRESHOLD = 0.3
# threshold for SAM's IOU score (confidence score for each mask), removing low quality predictions
cfg.MODEL.OpenWorldSAM2.TEST.IOU_THRESHOLD = 0.8
```

### Semantic segmentation (`semantic_inference.py`)

Generates class-wise logits conditioned on the prompts and renders a semantic map overlay. On ADE20K-150 for reference:

```bash
python demo/semantic_inference.py \
  --config-file configs/coco/panoptic-segmentation/Open-World-SAM2-CrossAttention.yaml \
  --weights "checkpoints/model_final.pth" \
  --image demo/images/giraffe.jpg \
  --prompts "sky" "tree" "building" "giraffe" "rock" "grass" \
  --output demo/outputs/giraffe_semantic.png
```

### Panoptic segmentation (`panoptic_inference.py`)

Runs the panoptic head, returning `(panoptic_seg, segments_info)` and saving the rendered mask. On COCO for reference:

```bash
python demo/panoptic_inference.py \
  --config-file configs/coco/panoptic-segmentation/Open-World-SAM2-CrossAttention.yaml \
  --weights "checkpoints/model_final.pth" \
  --image demo/images/giraffe.jpg \
  --prompts "sky" "tree" "building" "giraffe" "rock" "grass" \
  --output demo/outputs/giraffe_panoptic.png
```

### Referring expression segmentation (`referring_inference.py`)

Accepts arbitrary natural-language expressions and predicts the corresponding mask for each prompt.

```bash
python demo/referring_inference.py \
  --config-file configs/refcoco/Open-World-SAM2-CrossAttention.yaml \
  --weights "checkpoints/model_final_refcocog.pth" \
  --image demo/images/zebra.jpg \
  --prompts "zebra top left" \
  --output demo/outputs/zebra_referring.png
```

```bash
python demo/referring_inference.py \
  --config-file configs/refcoco/Open-World-SAM2-CrossAttention.yaml \
  --weights "checkpoints/model_final_refcocog.pth" \
  --image demo/images/dog.jpg \
  --prompts "a dog with a blue collar on a bed" "a dog with brown fur, with its head up, laying on a gray sheet" \
  --output demo/outputs/dog_referring.png
```

## Examples

Example on instance, semantic, and ponoptic segmentation on COCO-style images:

| Image                        | Instance                                              | Semantic                               | Panoptic                               |
| ---------------------------- | ----------------------------------------------------- | -------------------------------------- | -------------------------------------- |
| ![](./images/giraffe.jpg) | ![](./outputs/giraffe_instance__nms0.5_iou0.7.png) | ![](./outputs/giraffe_semantic.png) | ![](./outputs/giraffe_panoptic.png) |

Examples on instance and referring expression segmentation on common objects:

| Image                      | Instance                                           | Referring Expression                  |
| -------------------------- | -------------------------------------------------- | ------------------------------------- |
| ![](./images/zebra.jpg)   | ![](./outputs/zebra_instance_nms0.3_iou0.8.png)   | ![](./outputs/zebra_referring.png)   |
| ![](./images/dog.jpg)   | ![](./outputs/dog_instance_nms0.3_iou0.9.png)   | ![](./outputs/dog_referring.png)   |
| ![](./images/donut.jpg) | ![](./outputs/donut_instance_nms0.2_iou0.9.png) | ![](./outputs/donut_referring.png) |
| ![](./images/cake.jpg)  | ![](./outputs/cake_instance_nms0.3_iou0.9.png)  | ![](./outputs/cake_referring.png)  |

Examples on referring expression segmentation on long-tail objects and out-of-distribution images:

|                  | Image                          | Referring Expression                    |
| ---------------- | ------------------------------ | --------------------------------------- |
| OOD image        | ![](./images/xray_hand.jpg) | ![](./outputs/xray_referring.png)    |
| Long-tail object | ![](./images/ukulele.jpg)   | ![](./outputs/ukulele_referring.png) |
