# Original sources:
# - https://github.com/SHTUPLUS/PySGG/blob/main/pysgg/data/datasets/evaluation/oi/oi_evaluation.py
# - https://github.com/SHTUPLUS/PySGG/blob/main/pysgg/data/datasets/evaluation/coco/coco_eval.py

"""
Written by Ji Zhang, 2019
Some functions are adapted from Rowan Zellers
Original source:
https://github.com/rowanz/neural-motifs/blob/master/lib/evaluation/sg_eval.py
"""
from collections import OrderedDict, defaultdict
from functools import reduce

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from ..pytorch_misc import argsort_desc
from .ap_eval_rel import ap_eval, prepare_mAP_dets
from .sg_eval import _compute_pred_matches

np.set_printoptions(precision=3)


def _xyxy_to_xywh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]


# https://github.com/SHTUPLUS/PySGG/blob/a63942a076932b3756a477cf8919c3b74cd36207/pysgg/data/datasets/evaluation/coco/coco_eval.py#L327
class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


# https://github.com/SHTUPLUS/PySGG/blob/a63942a076932b3756a477cf8919c3b74cd36207/pysgg/data/datasets/evaluation/oi/oi_evaluation.py#L146
def eval_rel_results(all_results, predicate_cls_list, result_str=""):
    topk = 100

    prd_k = 2
    all_gt_cnt = 0
    recalls_per_img = {1: [], 5: [], 10: [], 20: [], 50: [], 100: []}
    recalls = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}

    topk_dets = []
    for im_i, res in enumerate(tqdm(all_results)):

        # in oi_all_rel some images have no dets
        if res["pred_scores"] is None:
            det_boxes_s_top = np.zeros((0, 4), dtype=np.float32)
            det_boxes_o_top = np.zeros((0, 4), dtype=np.float32)
            det_labels_s_top = np.zeros(0, dtype=np.int32)
            det_labels_p_top = np.zeros(0, dtype=np.int32)
            det_labels_o_top = np.zeros(0, dtype=np.int32)
            det_scores_top = np.zeros(0, dtype=np.float32)

        else:
            det_boxes_sbj = res["sbj_boxes"]  # (#num_rel, 4)
            det_boxes_obj = res["obj_boxes"]  # (#num_rel, 4)
            det_labels_sbj = res["sbj_labels"]  # (#num_rel,)
            det_labels_obj = res["obj_labels"]  # (#num_rel,)
            det_scores_sbj = res["sbj_scores"]  # (#num_rel,)
            det_scores_obj = res["obj_scores"]  # (#num_rel,)
            if "pred_scores_ttl" in res:
                det_scores_prd = res["pred_scores_ttl"]  # [:, 1:]
            else:
                det_scores_prd = res[
                    "pred_scores"
                ]  # [:, 1:] # N x C (the prediction score of each categories)

            det_labels_prd = np.argsort(
                -det_scores_prd
            )  # N x C (the prediction labels sort by prediction score)
            det_scores_prd = -np.sort(
                -det_scores_prd, axis=1
            )  # N x C (the prediction scores sort by prediction score)

            # filtering the results by the productiong of prediction score of subject object and predicates
            det_scores_so = det_scores_sbj * det_scores_obj  # N
            det_scores_spo = (
                det_scores_so[:, None] * det_scores_prd[:, :prd_k]
            )  # N x prd_K
            # (take top k predicates prediction of each pairs as final prediction, approximation of non-graph constrain setting)

            det_scores_inds = argsort_desc(det_scores_spo)[:topk]  # topk x 2
            # selected the topk score prediction from the N x prd_k predictions
            # first dim: which pair prediction. second dim: which cate prediction from this pair

            # take out the correspond tops relationship predation scores and pair boxes and their labels.
            det_scores_top = det_scores_spo[
                det_scores_inds[:, 0], det_scores_inds[:, 1]
            ]
            det_boxes_so_top = np.hstack(
                (
                    det_boxes_sbj[det_scores_inds[:, 0]],
                    det_boxes_obj[det_scores_inds[:, 0]],
                )
            )
            det_labels_p_top = det_labels_prd[
                det_scores_inds[:, 0], det_scores_inds[:, 1]
            ]
            det_labels_spo_top = np.vstack(
                (
                    det_labels_sbj[det_scores_inds[:, 0]],
                    det_labels_p_top,
                    det_labels_obj[det_scores_inds[:, 0]],
                )
            ).transpose()

            # filter the very low prediction scores relationship prediction
            cand_inds = np.where(det_scores_top > 0.00001)[0]
            det_boxes_so_top = det_boxes_so_top[cand_inds]
            det_labels_spo_top = det_labels_spo_top[cand_inds]
            det_scores_top = det_scores_top[cand_inds]

            det_boxes_s_top = det_boxes_so_top[:, :4]
            det_boxes_o_top = det_boxes_so_top[:, 4:]
            det_labels_s_top = det_labels_spo_top[:, 0]
            det_labels_p_top = det_labels_spo_top[:, 1]
            det_labels_o_top = det_labels_spo_top[:, 2]

        topk_dets.append(
            dict(
                image=im_i,
                det_boxes_s_top=det_boxes_s_top,
                det_boxes_o_top=det_boxes_o_top,
                det_labels_s_top=det_labels_s_top,
                det_labels_p_top=det_labels_p_top,
                det_labels_o_top=det_labels_o_top,
                det_scores_top=det_scores_top,
            )
        )

        gt_boxes_sbj = res["gt_sbj_boxes"]  # (#num_gt, 4)
        gt_boxes_obj = res["gt_obj_boxes"]  # (#num_gt, 4)
        gt_labels_sbj = res["gt_sbj_labels"]  # (#num_gt,)
        gt_labels_obj = res["gt_obj_labels"]  # (#num_gt,)
        gt_labels_prd = res["gt_prd_labels"]  # (#num_gt,)

        gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
        gt_labels_spo = np.vstack(
            (gt_labels_sbj, gt_labels_prd, gt_labels_obj)
        ).transpose()
        # Compute recall. It's most efficient to match once and then do recall after
        # det_boxes_so_top is (#num_rel, 8)
        # det_labels_spo_top is (#num_rel, 3)
        pred_to_gt = _compute_pred_matches(
            gt_labels_spo, det_labels_spo_top, gt_boxes_so, det_boxes_so_top, 0.5
        )

        # perimage recall
        for k in recalls_per_img:
            if len(pred_to_gt):
                match = reduce(np.union1d, pred_to_gt[:k])
            else:
                match = []
            rec_i = float(len(match)) / float(
                gt_labels_spo.shape[0] + 1e-12
            )  # in case there is no gt
            recalls_per_img[k].append(rec_i)

        # all dataset recall
        all_gt_cnt += gt_labels_spo.shape[0]
        for k in recalls:
            if len(pred_to_gt):
                match = reduce(np.union1d, pred_to_gt[:k])
            else:
                match = []
            recalls[k] += len(match)

        topk_dets[-1].update(
            dict(
                gt_boxes_sbj=gt_boxes_sbj,
                gt_boxes_obj=gt_boxes_obj,
                gt_labels_sbj=gt_labels_sbj,
                gt_labels_obj=gt_labels_obj,
                gt_labels_prd=gt_labels_prd,
            )
        )

    rel_prd_cats = predicate_cls_list  # [1:]
    for k in recalls_per_img.keys():
        recalls_per_img[k] = np.mean(recalls_per_img[k])

    for k in recalls:
        recalls[k] = float(recalls[k]) / (float(all_gt_cnt) + 1e-12)

    # prepare dets for each class
    cls_image_ids, cls_dets, cls_gts, npos = prepare_mAP_dets(
        topk_dets, len(rel_prd_cats)
    )
    all_npos = sum(npos)

    rel_mAP = 0.0
    w_rel_mAP = 0.0
    ap_str = ""
    per_class_res = ""
    for c in range(len(rel_prd_cats)):
        rec, prec, ap = ap_eval(
            cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], True
        )
        weighted_ap = ap * float(npos[c]) / float(all_npos)
        w_rel_mAP += weighted_ap
        rel_mAP += ap
        ap_str += "{:.2f}, ".format(100 * ap)
        per_class_res += "{}: {:.3f} / {:.3f} ({:.6f}), ".format(
            rel_prd_cats[c],
            100 * ap,
            100 * weighted_ap,
            float(npos[c]) / float(all_npos),
        )

    rel_mAP /= len(rel_prd_cats)
    result_str += "\nrel mAP: {:.2f}, weighted rel mAP: {:.2f}\n".format(
        100 * rel_mAP, 100 * w_rel_mAP
    )
    result_str += "rel AP perclass: AP/ weighted-AP (recall)\n"
    result_str += per_class_res + "\n\n"
    phr_mAP = 0.0
    w_phr_mAP = 0.0
    ap_str = ""

    per_class_res = ""
    for c in range(len(rel_prd_cats)):
        rec, prec, ap = ap_eval(
            cls_image_ids[c], cls_dets[c], cls_gts[c], npos[c], False
        )
        weighted_ap = ap * float(npos[c]) / float(all_npos)
        w_phr_mAP += weighted_ap
        phr_mAP += ap
        ap_str += "{:.2f}, ".format(100 * ap)
        per_class_res += "{}: {:.3f} / {:.3f} ({:.6f}), ".format(
            rel_prd_cats[c],
            100 * ap,
            100 * weighted_ap,
            float(npos[c]) / float(all_npos),
        )

    phr_mAP /= len(rel_prd_cats)
    result_str += "\nphr mAP: {:.2f}, weighted phr mAP: {:.2f}\n".format(
        100 * phr_mAP, 100 * w_phr_mAP
    )
    result_str += "phr AP perclass: AP/ weighted-AP (recall)\n"
    result_str += per_class_res + "\n\n"

    r50 = recalls[50]
    score = w_rel_mAP * 0.4 + w_phr_mAP * 0.4 + r50 * 0.2
    return {
        "w_rel_mAP": w_rel_mAP,
        "w_phr_mAP": w_phr_mAP,
        "microR@50": r50,
        "score": score,
    }


# https://github.com/SHTUPLUS/PySGG/blob/a63942a076932b3756a477cf8919c3b74cd36207/pysgg/data/datasets/evaluation/oi/oi_evaluation.py#L25
def eval_entites_detection(all_results, ind_to_classes):
    # create a Coco-like object that we can use to evaluate detection!
    anns = []
    result_str = ""

    for image_id, _result in enumerate(all_results):
        labels = _result["gt_class"].tolist()  # integer
        boxes = _result["gt_boxes"].tolist()  # xyxy
        for cls, box in zip(labels, boxes):
            anns.append(
                {
                    "area": (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    "bbox": [
                        box[0],
                        box[1],
                        box[2] - box[0] + 1,
                        box[3] - box[1] + 1,
                    ],  # xywh
                    "category_id": cls,
                    "id": len(anns),
                    "image_id": image_id,
                    "iscrowd": 0,
                }
            )
    fauxcoco = COCO()
    fauxcoco.dataset = {
        "info": {"description": "use coco script for oi detection evaluation"},
        "images": [{"id": i} for i in range(len(all_results))],
        "categories": [
            {"supercategory": "person", "id": i, "name": name}
            for i, name in enumerate(ind_to_classes)
            if name != "__background__"
        ],
        "annotations": anns,
    }
    fauxcoco.createIndex()

    # format predictions to coco-like
    cocolike_predictions = []
    for image_id, _result in enumerate(all_results):
        box = _result["pred_boxes"]
        label = _result["pred_class"]
        score = _result["pred_cls_scores"]
        box = [_xyxy_to_xywh(_box) for _box in box]  # xywh
        image_id = np.asarray([image_id] * len(box))
        cocolike_predictions.append(np.column_stack((image_id, box, score, label)))
        # logger.info(cocolike_predictions)
    cocolike_predictions = np.concatenate(cocolike_predictions, 0)

    res = fauxcoco.loadRes(cocolike_predictions)
    coco_eval = COCOeval(fauxcoco, res, "bbox")
    coco_eval.params.imgIds = list(range(len(all_results)))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_res = COCOResults("bbox")
    coco_res.update(coco_eval)
    mAp = coco_eval.stats[1]

    def get_coco_eval(coco_eval, iouThr, eval_type, maxDets=-1, areaRng="all"):
        p = coco_eval.params

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        if maxDets == -1:
            max_range_i = np.argmax(p.maxDets)
            mind = [
                max_range_i,
            ]
        else:
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if eval_type == "precision":
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval["precision"]
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        elif eval_type == "recall":
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval["recall"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        else:
            raise ValueError("Invalid eval metrics")
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return p.maxDets[mind[-1]], mean_s

    coco_res_to_save = {}
    for key, value in coco_res.results.items():
        for evl_name, eval_val in value.items():
            coco_res_to_save[f"{key}/{evl_name}"] = eval_val
    print(coco_res_to_save)

    result_str += "Detection evaluation mAp=%.4f\n" % mAp
    result_str += "recall@%d IOU:0.5 %.4f\n" % get_coco_eval(coco_eval, 0.5, "recall")
    result_str += "=" * 100 + "\n"
    avg_metrics = mAp
    print(result_str)
    return coco_res_to_save


class OICocoEvaluator:
    def __init__(self, predicate_cls_list, ind_to_classes):
        self.predicate_cls_list = predicate_cls_list
        self.ind_to_classes = ind_to_classes
        self.all_result = []

    def __call__(self, gt_entry, pred_entry):
        for _gt, _pred in zip(gt_entry, pred_entry.values()):
            gt_boxes = _gt["boxes"]
            gt_class = _gt["class_labels"]
            result_dict = defaultdict(list)
            result_dict["gt_boxes"].extend(gt_boxes.cpu())
            result_dict["gt_class"].extend(gt_class.cpu())

            pred_boxes = _pred["boxes"]
            pred_class = _pred["labels"]
            pred_score = _pred["scores"]
            result_dict["pred_boxes"].extend(pred_boxes.cpu())
            result_dict["pred_class"].extend(pred_class.cpu())
            result_dict["pred_cls_scores"].extend(pred_score.cpu())

            result_dict = dict(result_dict)
            for key, value in result_dict.items():
                result_dict[key] = np.array(value)
            self.all_result.append(result_dict)

    def aggregate_metrics(self):
        log_dict = {}
        log_dict.update(eval_entites_detection(self.all_result, self.ind_to_classes))
        return log_dict


class OIEvaluator:
    def __init__(self, predicate_cls_list, ind_to_classes):
        self.predicate_cls_list = predicate_cls_list
        self.ind_to_classes = ind_to_classes
        self.all_result = []

    def __call__(
        self,
        gt_entry,
        pred_entry,
    ):
        gt_boxes = gt_entry["gt_boxes"]
        gt_class = gt_entry["gt_classes"]
        result_dict = defaultdict(list)
        result_dict["gt_boxes"].extend(gt_boxes)
        result_dict["gt_class"].extend(gt_class)
        for _sbj, _obj, _rel in gt_entry["gt_relations"]:
            result_dict["gt_sbj_boxes"].append(gt_boxes[_sbj])
            result_dict["gt_obj_boxes"].append(gt_boxes[_obj])
            result_dict["gt_sbj_labels"].append(gt_class[_sbj])
            result_dict["gt_obj_labels"].append(gt_class[_obj])
            result_dict["gt_prd_labels"].append(_rel)

        pred_boxes = pred_entry["pred_boxes"]
        pred_class = pred_entry["pred_classes"]
        pred_score = pred_entry["obj_scores"]
        result_dict["pred_boxes"].extend(pred_boxes)
        result_dict["pred_class"].extend(pred_class)
        result_dict["pred_cls_scores"].extend(pred_score)
        for _sbj, _obj in pred_entry["sbj_obj_inds"]:
            result_dict["sbj_boxes"].append(pred_boxes[_sbj])
            result_dict["obj_boxes"].append(pred_boxes[_obj])
            result_dict["sbj_labels"].append(pred_class[_sbj])
            result_dict["obj_labels"].append(pred_class[_obj])
            result_dict["sbj_scores"].append(pred_score[_sbj])
            result_dict["obj_scores"].append(pred_score[_obj])
        result_dict["pred_scores"] = pred_entry["pred_scores"]  # N,
        result_dict = dict(result_dict)
        for key, value in result_dict.items():
            result_dict[key] = np.array(value)
        self.all_result.append(result_dict)

    def aggregate_metrics(self):
        log_dict = {}
        log_dict.update(eval_entites_detection(self.all_result, self.ind_to_classes))
        log_dict.update(eval_rel_results(self.all_result, self.predicate_cls_list))
        return log_dict
