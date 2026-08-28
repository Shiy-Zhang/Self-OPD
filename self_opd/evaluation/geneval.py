"""Local GenEval strict and continuous evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


COLORS = (
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"
)


def _intersection_over_union(box_a: np.ndarray, box_b: np.ndarray) -> float:
    def area(box):
        return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

    intersection = area(
        [
            max(box_a[0], box_b[0]),
            max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]),
            min(box_a[3], box_b[3]),
        ]
    )
    union = area(box_a) + area(box_b) - intersection
    return float(intersection / union) if union else 0.0


def _relative_positions(obj_a, obj_b, threshold: float) -> set[str]:
    boxes = np.asarray([obj_a[0][:4], obj_b[0][:4]]).reshape(2, 2, 2)
    centers = boxes.mean(axis=-2)
    dimensions = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = centers[0] - centers[1]
    revised = np.maximum(np.abs(offset) - threshold * dimensions.sum(axis=0), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3) or np.linalg.norm(offset) == 0:
        return set()
    dx, dy = revised / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


class GenevalScorer:
    """Score GenEval metadata using MMDetection and OpenCLIP."""

    detection_threshold = 0.3
    counting_threshold = 0.9
    max_objects = 16
    nms_threshold = 1.0
    position_threshold = 0.1

    def __init__(
        self,
        detector_config: str | Path,
        detector_checkpoint: str | Path,
        classnames: str | Path,
        device: str = "cuda",
        clip_architecture: str = "ViT-L-14",
        clip_pretrained: str = "openai",
    ) -> None:
        from mmdet.apis import init_detector
        import open_clip

        for path in (detector_config, detector_checkpoint, classnames):
            if not Path(path).expanduser().is_file():
                raise FileNotFoundError(f"Required GenEval file not found: {path}")
        self.device = device
        self.detector = init_detector(
            str(Path(detector_config).expanduser()),
            str(Path(detector_checkpoint).expanduser()),
            device=device,
        )
        with Path(classnames).expanduser().open(encoding="utf-8") as handle:
            self.classnames = [line.strip() for line in handle if line.strip()]
        self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
            clip_architecture, pretrained=clip_pretrained, device=device
        )
        self.clip_tokenizer = open_clip.get_tokenizer(clip_architecture)
        self.clip_model.eval()
        self._color_cache: dict[str, torch.Tensor] = {}

    def _detect(self, images: list[np.ndarray]):
        from mmdet.apis import inference_detector

        parsed = []
        for image in images:
            result = inference_detector(self.detector, image)
            if isinstance(result, tuple):
                boxes, masks = result
                parsed.append((boxes, masks))
                continue

            instances = result.pred_instances.cpu()
            scores = instances.scores.numpy()
            labels = instances.labels.numpy()
            boxes = instances.bboxes.numpy()
            masks = instances.masks.numpy() if hasattr(instances, "masks") else None
            boxes_by_class = [np.empty((0, 5)) for _ in self.classnames]
            masks_by_class: list[Any] = [None] * len(self.classnames)
            for class_index in range(len(self.classnames)):
                indices = np.where(labels == class_index)[0]
                if len(indices):
                    boxes_by_class[class_index] = np.concatenate(
                        [boxes[indices], scores[indices, None]], axis=1
                    )
                    if masks is not None:
                        masks_by_class[class_index] = masks[indices]
            parsed.append((boxes_by_class, masks_by_class))
        return parsed

    @torch.inference_mode()
    def _color_classifier(self, classname: str) -> torch.Tensor:
        if classname in self._color_cache:
            return self._color_cache[classname]
        templates = (
            "a photo of a {color} " + classname,
            "a photo of a {color}-colored " + classname,
            "a photo of a {color} object",
        )
        features = []
        for color in COLORS:
            tokens = self.clip_tokenizer(
                [template.format(color=color) for template in templates]
            ).to(self.device)
            encoded = self.clip_model.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            features.append(encoded.mean(dim=0))
        classifier = torch.stack(features)
        classifier = classifier / classifier.norm(dim=-1, keepdim=True)
        self._color_cache[classname] = classifier
        return classifier

    @torch.inference_mode()
    def _classify_colors(self, image: Image.Image, objects, classname: str) -> list[str]:
        background = Image.new("RGB", image.size, color="#999")
        crops = []
        for box, mask in objects:
            composed = image
            if mask is not None:
                mask_array = np.asarray(mask).astype(np.uint8) * 255
                composed = Image.composite(image, background, Image.fromarray(mask_array, mode="L"))
            crops.append(self.clip_transform(composed.crop(box[:4].astype(int).tolist())))
        if not crops:
            return []
        features = self.clip_model.encode_image(torch.stack(crops).to(self.device))
        features = features / features.norm(dim=-1, keepdim=True)
        indices = (features @ self._color_classifier(classname).T).argmax(dim=1)
        return [COLORS[index] for index in indices.cpu().tolist()]

    def _objects(self, boxes_by_class, masks_by_class, tag: str):
        threshold = self.counting_threshold if tag == "counting" else self.detection_threshold
        detected = {}
        for class_index, classname in enumerate(self.classnames):
            boxes = np.asarray(boxes_by_class[class_index])
            if not len(boxes):
                continue
            candidates = np.argsort(boxes[:, 4])[::-1]
            candidates = candidates[boxes[candidates, 4] > threshold][: self.max_objects]
            kept_indices = []
            kept = []
            class_masks = masks_by_class[class_index] if masks_by_class is not None else None
            for index in candidates.tolist():
                if self.nms_threshold == 1 or all(
                    _intersection_over_union(boxes[index], boxes[other]) < self.nms_threshold
                    for other in kept_indices
                ):
                    mask = class_masks[index] if class_masks is not None else None
                    kept.append((boxes[index], mask))
                    kept_indices.append(index)
            if kept:
                detected[classname] = kept
        return detected

    def _evaluate(self, image: Image.Image, objects: dict, metadata: dict) -> tuple[float, float]:
        exact = True
        partial_scores = []
        matched_groups = []
        for requirement in metadata.get("include", []):
            classname = requirement["class"]
            expected = requirement["count"]
            found = objects.get(classname, [])
            matched = len(found) == expected
            partial_scores.append(1.0 - abs(expected - len(found)) / max(expected, 1))
            if not matched:
                exact = False
                if "color" in requirement or "position" in requirement:
                    partial_scores.append(0.0)
            elif "color" in requirement:
                colors = self._classify_colors(image, found, classname)
                correct_colors = colors.count(requirement["color"])
                partial_scores.append(1.0 - abs(expected - correct_colors) / max(expected, 1))
                if correct_colors != expected:
                    exact = matched = False
            if "position" in requirement and matched:
                relation, target_group = requirement["position"]
                target_objects = matched_groups[target_group]
                position_ok = target_objects is not None and all(
                    relation in _relative_positions(obj, target, self.position_threshold)
                    for obj in found
                    for target in target_objects
                )
                partial_scores.append(float(position_ok))
                if not position_ok:
                    exact = matched = False
            matched_groups.append(found if matched else None)

        for requirement in metadata.get("exclude", []):
            if len(objects.get(requirement["class"], [])) >= requirement["count"]:
                exact = False
        continuous = float(np.mean(partial_scores)) if partial_scores else 0.0
        return float(exact), continuous

    @torch.inference_mode()
    def __call__(
        self, images: list[Image.Image], metadata: list[dict]
    ) -> tuple[list[float], list[float]]:
        if len(images) != len(metadata):
            raise ValueError("Images and GenEval metadata must have the same length")
        pil_images = [ImageOps.exif_transpose(image.convert("RGB")) for image in images]
        detections = self._detect([np.asarray(image) for image in pil_images])
        strict_scores, continuous_scores = [], []
        for image, result, item in zip(pil_images, detections, metadata):
            objects = self._objects(*result, item.get("tag", ""))
            strict, continuous = self._evaluate(image, objects, item)
            strict_scores.append(strict)
            continuous_scores.append(continuous)
        return strict_scores, continuous_scores


def aggregate_geneval(
    metadata: list[dict], strict: list[float], continuous: list[float]
) -> dict[str, Any]:
    if not (len(metadata) == len(strict) == len(continuous)):
        raise ValueError("GenEval result lengths do not match")
    by_tag: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"strict": [], "continuous": []}
    )
    for item, strict_score, continuous_score in zip(metadata, strict, continuous):
        tag = item.get("tag", "unknown")
        by_tag[tag]["strict"].append(strict_score)
        by_tag[tag]["continuous"].append(continuous_score)
    return {
        "strict": float(np.mean(strict)),
        "continuous": float(np.mean(continuous)),
        "count": len(strict),
        "per_tag": {
            tag: {
                "strict": float(np.mean(values["strict"])),
                "continuous": float(np.mean(values["continuous"])),
                "count": len(values["strict"]),
            }
            for tag, values in sorted(by_tag.items())
        },
    }
