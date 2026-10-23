"""Load COCO files and separate box supervision from image-level labels."""

import json
from collections import defaultdict
from pathlib import Path


def _read_files(filenames):
    if filenames is None:
        return []
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    datasets = []
    for filename in filenames:
        filename = Path(filename)
        with filename.open(encoding="utf-8") as handle:
            datasets.append((filename, json.load(handle)))
    return datasets


def _det_class_split(datasets, class_ids, data_dir):
    dataset_det, dataset_class = [], []
    image_id = 0
    for filename, coco in datasets:
        category_ids = {
            category["id"]: class_ids[category["name"]]
            for category in coco["categories"]
        }
        annotations = defaultdict(list)
        for annotation in coco.get("annotations", []):
            annotations[annotation["image_id"]].append(annotation)

        for image in coco["images"]:
            boxes, labels = [], []
            for annotation in annotations[image["id"]]:
                category_id = category_ids[annotation["category_id"]]
                if annotation.get("bbox") is not None and len(annotation["bbox"]) == 4:
                    box = dict(annotation, category_id=category_id, bbox_mode=1)
                    box.setdefault("iscrowd", 0)
                    boxes.append(box)
                else:
                    labels.append(category_id)
            for field in ("category_ids", "pos_category_ids"):
                labels.extend(category_ids[label] for label in image.get(field, []))
            if "category_id" in image:
                labels.append(category_ids[image["category_id"]])

            record = {
                "image_id": image_id,
                "file_name": str(data_dir / image["file_name"]),
                "width": image["width"],
                "height": image["height"],
            }
            image_id += 1
            if boxes or not labels:
                dataset_det.append(dict(
                    record, annotations=boxes, pos_category_ids=[],
                    ann_type="box", dataset_source=0,
                ))
            if labels:
                dataset_class.append(dict(
                    record, annotations=[], pos_category_ids=sorted(set(labels)),
                    ann_type="image", dataset_source=1,
                ))
    return {"detection": dataset_det, "classification": dataset_class}


def load_coco_datasets(train, test=None, *, data_dir):
    """Accept one COCO path or a list of paths for each split.

    Relative image file names resolve directly against data_dir; no Images
    subdirectory is added. Absolute file names are preserved.
    Class names align category IDs across files. Images containing both kinds
    of annotations contribute to both groups; unannotated images are detection
    backgrounds. Returned records use Detectron2's zero-based class IDs.
    """
    data_dir = Path(data_dir)
    train_files, test_files = _read_files(train), _read_files(test)
    class_ids = {}
    for _, coco in train_files + test_files:
        for category in sorted(coco["categories"], key=lambda item: item["id"]):
            if category["name"] not in class_ids:
                class_ids[category["name"]] = len(class_ids)
    return {
        "classes": list(class_ids),
        "train": _det_class_split(train_files, class_ids, data_dir),
        "test": _det_class_split(test_files, class_ids, data_dir),
    }
