import glob
import json
import os
from typing import Dict, List

from datasets import Dataset, Features
from datasets import Image as HFImage
from datasets import Value
from PIL import Image
from tqdm import tqdm

# we need to define the features ourselves
features = Features(
    {
        "name": Value(dtype="string"),
        "rgb": HFImage(decode=True),
        "depth": HFImage(decode=True),
        "gt": HFImage(decode=True),
    }
)


def push_to_hub(
    dataset_path: str, split: str, rgb_ext="jpg", depth_ext="png", gt_ext="png"
):
    rgb_paths = glob.glob(f"{dataset_path}/RGB/*.{rgb_ext}")
    depth_paths = glob.glob(f"{dataset_path}/depths/*.{depth_ext}")
    gt_paths = glob.glob(f"{dataset_path}/GT/*.{gt_ext}")

    metadata: Dict[str, List] = dict(name=[], rgb=[], depth=[], gt=[])

    for rgb_path in tqdm(rgb_paths):
        name = os.path.basename(rgb_path).split(".")[0]
        depth_path = f"{dataset_path}/depths/{name}.{depth_ext}"
        gt_path = f"{dataset_path}/GT/{name}.{gt_ext}"

        if depth_path not in depth_paths:
            raise Exception(f"Depth image not found: {depth_path}")
        if gt_path not in gt_paths:
            raise Exception(f"GT image not found: {gt_path}")

        metadata["rgb"].append(Image.open(rgb_path))
        metadata["depth"].append(Image.open(depth_path))
        metadata["gt"].append(Image.open(gt_path))
        metadata["name"].append(name)

    dataset = Dataset.from_dict(metadata, split=split)
    dataset.push_to_hub("RGBD-SOD/COME15K", split=split)


push_to_hub("rgbdsod_datasets/train", "train")
push_to_hub("rgbdsod_datasets/test/COME-E", "validation")
push_to_hub("rgbdsod_datasets/test/COME-H", "test")
