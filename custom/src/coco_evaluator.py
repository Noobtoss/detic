from copy import deepcopy

from detectron2.evaluation.coco_evaluation import COCOEvaluator as _COCOEvaluator
from detectron2.evaluation.coco_evaluation import COCOevalMaxDets


# Only listed metrics are returned: original name -> output name.
COCO_METRICS = {
    "AP": "mAP50-95",
    "AP_agnostic": "mAP50-95_agnostic",
}


class COCOEvaluator(_COCOEvaluator):
    def evaluate(self, img_ids=None):
        results = super().evaluate(img_ids=img_ids)
        return {
            task: {
                COCO_METRICS[key]: value
                for key, value in metrics.items()
                if key in COCO_METRICS
            }
            for task, metrics in results.items()
        }

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)
        if iou_type != "bbox":
            return results

        names = ("AP_agnostic", "AP50_agnostic", "AP75_agnostic")
        if coco_eval is None:
            results.update({name: float("nan") for name in names})
            return results

        # Reuse mapped predictions and evaluation settings, ignoring class labels.
        agnostic = COCOevalMaxDets(coco_eval.cocoGt, coco_eval.cocoDt, "bbox")
        agnostic.params = deepcopy(coco_eval.params)
        agnostic.params.useCats = 0
        self._logger.info("Evaluating class-agnostic bounding-box AP ...")
        agnostic.evaluate()
        agnostic.accumulate()
        agnostic.summarize()
        results.update({
            name: float(value * 100) if value >= 0 else float("nan")
            for name, value in zip(names, agnostic.stats[:3])
        })
        return results
