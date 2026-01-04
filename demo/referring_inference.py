import argparse
import logging
import torch

import os
import sys
# Ensure repository root is available on sys.path when executed as a script.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo.inference_utils import (
    build_inference_inputs,
    load_model,
    prepare_image_inputs,
    setup_cfg,
)
from utils.visualizer import SegmentationResultVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="OpenWorldSAM2 Referring Expression Segmentation Inference")
    parser.add_argument("--config-file", required=True, help="Path to the config file")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--prompts",
        required=True,
        nargs="+",
        help="Referring expressions describing the target regions",
    )
    parser.add_argument("--weights", default=None, help="Path to model weights")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    parser.add_argument("--output", default="outputs/referring_result.png", help="Path to save the visualization")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER, help="Additional config options")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = setup_cfg(args.config_file, weights=args.weights, device=args.device, opts=args.opts)
    cfg.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.PANOPTIC_ON = False
    cfg.MODEL.OpenWorldSAM2.TEST.REFER_ON = True

    prompts = [p.strip() for p in args.prompts]
    category_ids = list(range(len(prompts)))

    model = load_model(cfg)
    image_bgr, sam_tensor, beit_tensor, height, width = prepare_image_inputs(args.image, cfg.INPUT.FORMAT)
    inputs = build_inference_inputs(sam_tensor, beit_tensor, height, width, prompts, category_ids)

    with torch.no_grad():
        outputs = model(inputs)[0]

    grounding_masks = outputs.get("grounding_mask")
    grounding_scores = outputs.get("grounding_scores")
    if grounding_masks is None:
        raise RuntimeError("Referring expression masks are missing from the model output.")

    visualizer = SegmentationResultVisualizer(metadata=None, input_format=cfg.INPUT.FORMAT)
    visualizer.save_referring_result(image_bgr, grounding_masks, prompts, args.output, scores=grounding_scores)
    logging.info("Saved referring expression segmentation result to %s", args.output)


if __name__ == "__main__":
    main()
