# Reference: https://github.com/MCG-NJU/Structured-Sparse-RCNN/blob/main/maskrcnn_benchmark/data/datasets/open_image.py

import json
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_cate_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, "r"))
    ind_to_predicates_cate = info["rel"]
    ind_to_entites_cate = info["obj"]

    predicate_to_ind = {idx: name for idx, name in enumerate(ind_to_predicates_cate)}
    entites_cate_to_ind = {idx: name for idx, name in enumerate(ind_to_entites_cate)}

    return (
        ind_to_entites_cate,
        ind_to_predicates_cate,
        entites_cate_to_ind,
        predicate_to_ind,
    )


class OIDetection(torch.utils.data.Dataset):
    def __init__(self, data_folder, feature_extractor, split, debug=False):
        self.annotation_file = f"{data_folder}/annotations/vrd-{split}-anno.json"
        self.img_dir = f"{data_folder}/images"
        self.cate_info_file = f"{data_folder}/annotations/categories_dict.json"
        self.targets = json.load(open(self.annotation_file, "r"))
        (
            self.ind_to_classes,
            self.rel_categories,
            self.classes_to_ind,
            self.predicates_to_ind,
        ) = load_cate_info(self.cate_info_file)
        self.feature_extractor = feature_extractor
        self.split = split
        self.debug = debug

    def __getitem__(self, idx):
        target = self.targets[idx]
        img = Image.open(f"{self.img_dir}/{target['img_fn']}.jpg").convert("RGB")
        coco_target = self.convert_to_coco_format(idx)
        if self.feature_extractor is not None:
            encoding = self.feature_extractor(
                images=img, annotations=coco_target, return_tensors="pt"
            )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        label = encoding["labels"][0]  # remove batch dimension
        return pixel_values, label

    def convert_to_coco_format(self, index):
        target = self.targets[index]
        bboxes = target["bbox"]
        annotation = []
        for i, bbox in enumerate(bboxes):
            bbox = (
                [bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1],
            )  # xyxy -> xywh
            _anno = {
                "segmentation": None,
                "area": None,
                "bbox": bbox,
                "iscrowd": 0,
                "image_id": index,
                "category_id": target["det_labels"][i],
            }
            annotation.append(_anno)
        return {"image_id": index, "annotations": annotation}

    def __len__(self):
        if self.debug and self.split == "train":
            return 5000
        else:
            return len(self.targets)


class OIDataset(OIDetection):
    def __init__(
        self,
        data_folder,
        feature_extractor=None,
        split="train",
        filter_duplicate_rels=True,
        filter_multiple_rels=False,
        num_object_queries=100,
        debug=False,
    ):
        super(OIDataset, self).__init__(data_folder, feature_extractor, split, debug)
        assert split in {"train", "val", "test"}
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == "train"
        self.filter_multiple_rels = filter_multiple_rels and split == "train"
        self.remove_tail_classes = False
        self.num_object_queries = num_object_queries

        self.categories = {
            i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))
        }

        if split == "train":
            self.targets = [
                target
                for target in self.targets
                if len(target["bbox"]) <= self.num_object_queries
            ]
            if filter_duplicate_rels:
                # choose one relation between same subject and object
                assert self.split == "train"
                for idx, target in enumerate(self.targets):
                    all_rel_sets = defaultdict(list)
                    for sbj, obj, rel in target["rel"]:
                        all_rel_sets[(sbj, obj, rel)].append(rel)
                    self.targets[idx]["rel"] = [
                        [k[0], k[1], v[0]] for k, v in all_rel_sets.items()
                    ]
        self.pre_compute_bbox = None

    def __getitem__(self, idx):
        target = self.targets[idx]
        img = Image.open(f"{self.img_dir}/{target['img_fn']}.jpg").convert("RGB")
        coco_target = self.convert_to_coco_format(idx)
        rel_list = target["rel"]
        if self.filter_multiple_rels:
            all_rel_sets = defaultdict(list)
            for sbj, obj, rel in rel_list:
                all_rel_sets[(sbj, obj)].append(rel)
            rel_list = [
                [k[0], k[1], np.random.choice(v)] for k, v in all_rel_sets.items()
            ]
        if self.feature_extractor is not None:
            encoding = self.feature_extractor(
                images=img, annotations=coco_target, return_tensors="pt"
            )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        label = encoding["labels"][0]  # remove batch dimension
        rel = np.array(rel_list)
        label["rel"] = self._get_rel_tensor(rel)
        return pixel_values, label

    def _get_rel_tensor(self, rel_tensor):
        indices = torch.tensor(rel_tensor).T

        rel = torch.zeros(
            [
                self.num_object_queries,
                self.num_object_queries,
                len(self.predicates_to_ind),
            ]
        )
        rel[indices[0, :], indices[1, :], indices[2, :]] = 1.0
        return rel


def oi_get_statistics(train_data, must_overlap=True):
    """save the initial data distribution for the frequency bias model

    Args:
        train_data ([type]): the self
        must_overlap (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.rel_categories)
    fg_matrix = np.zeros(
        (num_obj_classes + 1, num_obj_classes + 1, num_rel_classes), dtype=np.int64
    )
    for target in tqdm(train_data.targets):  # use GT not augmented one
        gt_classes = np.array(target["det_labels"])
        gt_relations = np.array(target["rel"])

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
    return fg_matrix
