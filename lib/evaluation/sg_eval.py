# Original source: https://github.com/yuweihao/KERN/blob/master/lib/evaluation/sg_eval.py

"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import math
from functools import reduce

import numpy as np

from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from lib.pytorch_misc import argsort_desc, intersect_2d

np.set_printoptions(precision=3)

MODES = ["sgdet"]


class BasicSceneGraphEvaluator:
    def __init__(self, mode, multiple_preds=False):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + "_recall"] = {20: [], 50: [], 100: []}
        self.multiple_preds = multiple_preds

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in MODES}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {
            m: cls(mode=m, multiple_preds=True, **kwargs) for m in ("preddet", "phrdet")
        }
        return evaluators

    def evaluate_scene_graph_entry(
        self, gt_entry, pred_scores, viz_dict=None, iou_thresh=0.5
    ):
        res = evaluate_from_dict(
            gt_entry,
            pred_scores,
            self.mode,
            self.result_dict,
            viz_dict=viz_dict,
            iou_thresh=iou_thresh,
            multiple_preds=self.multiple_preds,
        )
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        output = {}
        if self.multiple_preds:
            recall_mode = "recall without constraint"
        else:
            recall_mode = "recall with constraint"
        print(
            "======================"
            + self.mode
            + "  "
            + recall_mode
            + "============================"
        )
        for k, v in self.result_dict[self.mode + "_recall"].items():
            print("R@%i: %f" % (k, np.mean(v)))
            output["R@%i" % k] = np.mean(v)
        return output


def evaluate_from_dict(
    gt_entry,
    pred_entry,
    mode,
    result_dict,
    multiple_preds=False,
    viz_dict=None,
    **kwargs
):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict:
    :param viz_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry["gt_relations"]
    gt_boxes = gt_entry["gt_boxes"].astype(float)
    gt_classes = gt_entry["gt_classes"]

    pred_rel_inds = pred_entry["pred_rel_inds"]
    rel_scores = pred_entry["rel_scores"]
    if mode == "predcls":
        pred_boxes = gt_boxes
        pred_classes = gt_classes
        obj_scores = np.ones(gt_classes.shape[0])
    elif mode == "sgcls":
        pred_boxes = gt_boxes
        pred_classes = pred_entry["pred_classes"]
        obj_scores = pred_entry["obj_scores"]
    elif mode.startswith("sgdet") or mode == "phrdet":
        pred_boxes = pred_entry["pred_boxes"].astype(float)
        pred_classes = pred_entry["pred_classes"]
        obj_scores = pred_entry["obj_scores"]
    elif mode == "preddet":
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + "_recall"]:
                result_dict[mode + "_recall"][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores)
        rel_scores_sorted = np.column_stack(
            (pred_rel_inds[rel_scores_sorted[:, 0]], rel_scores_sorted[:, 1])
        )

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + "_recall"]:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + "_recall"][k].append(rec_i)
        return None, None, None
    else:
        raise ValueError("invalid mode")

    if multiple_preds:
        pred_rels = pred_rel_inds  # [pred_rels, 3(s,o,p)]
        predicate_scores = rel_scores  # [pred_rels]
    else:
        pred_rels = np.column_stack(
            (pred_rel_inds, rel_scores.argmax(1))
        )  # [pred_rels, 3(s,o,p)]
        predicate_scores = rel_scores.max(1)  # [pred_rels]

    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
        gt_rels,
        gt_boxes,
        gt_classes,
        pred_rels,
        pred_boxes,
        pred_classes,
        predicate_scores,
        obj_scores,
        phrdet=mode == "phrdet",
        **kwargs
    )

    for k in result_dict[mode + "_recall"]:
        match = reduce(np.union1d, pred_to_gt[:k])
        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + "_recall"][k].append(rec_i)
    return pred_to_gt, pred_5ples, rel_scores


def evaluate_recall(
    gt_rels,
    gt_boxes,
    gt_classes,
    pred_rels,
    pred_boxes,
    pred_classes,
    rel_scores=None,
    cls_scores=None,
    iou_thresh=0.5,
    phrdet=False,
):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
    """
    if pred_rels.size == 0:
        return [[]], np.zeros((0, 5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(
        gt_rels[:, 2], gt_rels[:, :2], gt_classes, gt_boxes
    )
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:, :2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,1])
    assert np.all(pred_rels[:, 2] >= 0)

    pred_triplets, pred_triplet_boxes, relation_scores = _triplet(
        pred_rels[:, 2],
        pred_rels[:, :2],
        pred_classes,
        pred_boxes,
        rel_scores,
        cls_scores,
    )

    scores_overall = relation_scores.prod(1)
    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print(
            "Somehow the relations weren't sorted properly: \n{}".format(scores_overall)
        )
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack(
        (
            pred_rels[:, :2],
            pred_triplets[:, [0, 2, 1]],
        )
    )

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(
    predicates, relations, classes, boxes, predicate_scores=None, class_scores=None
):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-1) predicates, corresponding to
                    each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-1), 2) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-1)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert predicates.shape[0] == relations.shape[0]

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack(
            (
                class_scores[relations[:, 0]],
                class_scores[relations[:, 1]],
                predicate_scores,
            )
        )

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(
    gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh, phrdet=False
):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(
        np.where(gt_has_match)[0],
        gt_boxes[gt_has_match],
        keeps[gt_has_match],
    ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate(
                (gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0
            )

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate(
                (box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1
            )

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def calculate_mR_from_evaluator_list(evaluator_list, mode, multiple_preds=False):
    all_rel_results = {}
    for (pred_id, pred_name, evaluator_rel) in evaluator_list:
        print("\n")
        print("relationship: ", pred_name)
        rel_results = evaluator_rel[mode].print_stats()
        all_rel_results[pred_name] = rel_results

    mean_recall = {}
    mR20 = 0.0
    mR50 = 0.0
    mR100 = 0.0
    for key, value in all_rel_results.items():
        if math.isnan(value["R@100"]):
            continue
        mR20 += value["R@20"]
        mR50 += value["R@50"]
        mR100 += value["R@100"]
    rel_num = len(evaluator_list)
    mR20 /= rel_num
    mR50 /= rel_num
    mR100 /= rel_num
    mean_recall["mR@20"] = mR20
    mean_recall["mR@50"] = mR50
    mean_recall["mR@100"] = mR100
    all_rel_results["mean_recall"] = mean_recall
    if multiple_preds:
        recall_mode = "mean recall without constraint"
    else:
        recall_mode = "mean recall with constraint"
    print("\n")
    print(
        "======================"
        + mode
        + "  "
        + recall_mode
        + "============================"
    )
    print("mR@20: ", mR20)
    print("mR@50: ", mR50)
    print("mR@100: ", mR100)
    return mean_recall
