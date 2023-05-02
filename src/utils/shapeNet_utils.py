import os
import json
from src.poses.utils import get_root_project

train_cats = [
    "airplane",
    "bench",
    "cabinet",
    "car",
    "chair",
    "display",
    "lamp",
    "loudspeaker",
    "rifle",
    "sofa",
    "table",
    "telephone",
    "watercraft",  # "vessel" in the paper
]

test_cats = [
    "bottle",
    "bus",
    "clock",
    "dishwasher",
    "guitar",
    "mug",
    "pistol",
    "skateboard",
    "train",
    "washer",
]


def get_shapeNet_mapping():
    root_repo = get_root_project()
    path_shapenet_id2cat = os.path.join(
        root_repo, "src/utils/shapeNet_id2cat_v2.json"
    )
    with open(path_shapenet_id2cat) as json_file:
        id2cat_mapping = json.load(json_file)
    # create inverse mapping
    cat2id_mapping = {}
    for key, value in id2cat_mapping.items():
        cat2id_mapping[value] = key
    return id2cat_mapping, cat2id_mapping
