import argparse
import logging
import os
import sys
import torch

# Ensure repository root is available on sys.path when executed as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo.inference_utils import (
    build_inference_inputs,
    get_metadata,
    load_model,
    prepare_image_inputs,
    resolve_category_ids,
    setup_cfg,
)
from utils.visualizer import SegmentationResultVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="OpenWorldSAM2 Panoptic Segmentation Inference")
    parser.add_argument("--config-file", required=True, help="Path to the config file")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--prompts",
        required=True,
        nargs="+",
        help="List of textual prompts referencing dataset categories",
    )
    parser.add_argument("--weights", default=None, help="Path to model weights")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    parser.add_argument("--output", default="outputs/panoptic_result.png", help="Path to save the visualization")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER, help="Additional config options")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = setup_cfg(args.config_file, weights=args.weights, device=args.device, opts=args.opts)
    cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON = True
    cfg.MODEL.OpenWorldSAM2.TEST.REFER_ON = False

    metadata = get_metadata(cfg)
    prompts = [p.strip() for p in args.prompts]
    category_ids = resolve_category_ids(prompts, metadata)

    model = load_model(cfg)
    image_bgr, sam_tensor, beit_tensor, height, width = prepare_image_inputs(args.image, cfg.INPUT.FORMAT)
    inputs = build_inference_inputs(sam_tensor, beit_tensor, height, width, prompts, category_ids)

    with torch.no_grad():
        outputs = model(inputs)[0]

    panoptic_output = outputs.get("panoptic_seg")
    if panoptic_output is None:
        raise RuntimeError("Panoptic segmentation output is missing.")

    visualizer = SegmentationResultVisualizer(metadata=metadata, input_format=cfg.INPUT.FORMAT)
    visualizer.save_panoptic_result(image_bgr, panoptic_output, args.output)
    logging.info("Saved panoptic segmentation result to %s", args.output)


if __name__ == "__main__":
    main()
