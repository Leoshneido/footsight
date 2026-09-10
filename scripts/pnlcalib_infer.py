# scripts/pnlcalib_infer.py
"""
GPL-2.0: this script imports PnLCalib (https://github.com/mguti97/PnLCalib)
internals directly and is bound by PnLCalib's license. Run only as an
isolated subprocess from footsight — never import this module.
See docs/superpowers/specs/2026-09-10-core-cv-pipeline-design.md and
MEMORY.md, 2026-09-10, for why this boundary exists.
"""
import argparse
import json
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
import inference as pnlcalib_inference
from inference import inference, projection_from_cam_params


def load_models(weights_kp, weights_line, device):
    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml"))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml"))

    model = get_cls_net(cfg)
    model.load_state_dict(torch.load(weights_kp, map_location=device))
    model.to(device)
    model.eval()

    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(weights_line, map_location=device))
    model_l.to(device)
    model_l.eval()

    return model, model_l


def compute_homography(image_bgr, model, model_l, device):
    # PnLCalib's inference() reads these as module-level globals, which its
    # own module only ever sets inside its `if __name__ == "__main__"` block
    # (see vendor/PnLCalib/inference.py). Since we import it as a library
    # function rather than running it as a script, we must set them here.
    pnlcalib_inference.device = device
    pnlcalib_inference.transform2 = T.Resize((540, 960))

    h, w = image_bgr.shape[:2]
    cam = FramebyFrameCalib(iwidth=w, iheight=h, denormalize=True)
    final_params_dict = inference(
        cam, image_bgr, model, model_l,
        kp_threshold=0.3434, line_threshold=0.7867, pnl_refine=True,
    )
    if final_params_dict is None:
        return None

    P = projection_from_cam_params(final_params_dict)  # 3x4: world(meters, centered) -> image pixels
    H_world_to_image = P[:, [0, 1, 3]]                 # players stand on the Z=0 plane -> drop the Z column
    H_image_to_world = np.linalg.inv(H_world_to_image)
    return H_image_to_world.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights-kp", required=True)
    parser.add_argument("--weights-line", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(json.dumps({"error": f"could not read image: {args.image}"}), file=sys.stderr)
        sys.exit(1)

    model, model_l = load_models(args.weights_kp, args.weights_line, args.device)
    homography = compute_homography(image_bgr, model, model_l, args.device)
    print(json.dumps({"homography": homography}))


if __name__ == "__main__":
    main()
