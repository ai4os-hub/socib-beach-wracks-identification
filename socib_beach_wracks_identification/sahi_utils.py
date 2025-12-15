import random

import cv2
import numpy as np


class SahiCustomResult:
    """
    Wrapper for SAHI prediction results to provide custom JSON output
    and plotting.
    """

    def __init__(self, sahi_result, original_image):
        self.sahi_result = sahi_result
        self.orig_img = original_image
        self.object_prediction_list = sahi_result.object_prediction_list

        self.colors = {}

    def _get_color(self, class_id):
        """Returns a fixed color for each class_id."""
        if class_id not in self.colors:
            random.seed(class_id)
            self.colors[class_id] = [random.randint(0, 255) for _ in range(3)]
        return self.colors[class_id]

    def to_json(self):
        """
        Generates the JSON output with bounding boxes and segmentation
        polygons.
        """
        predictions = []
        for pred in self.object_prediction_list:
            bbox = pred.bbox.to_xyxy()

            segmentation = {"x": [], "y": []}
            if pred.mask:
                mask_bool = pred.mask.bool_mask

                mask_uint8 = (mask_bool * 255).astype(np.uint8)

                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    c = max(contours, key=cv2.contourArea)

                    points = c.reshape(-1, 2)

                    segmentation = {
                        "x": points[:, 0].tolist(),
                        "y": points[:, 1].tolist(),
                    }

            predictions.append(
                {
                    "name": pred.category.name,
                    "class": pred.category.id,
                    "confidence": float(pred.score.value),
                    "box": {
                        "x1": int(bbox[0]),
                        "y1": int(bbox[1]),
                        "x2": int(bbox[2]),
                        "y2": int(bbox[3]),
                    },
                    "segments": segmentation,
                }
            )
        return predictions

    def plot(
        self,
        conf=True,
        labels=True,
        boxes=True,
        font_size=1.0,
        alpha=0.5,
        **kwargs,
    ):
        """
        Draws segmentation masks and boxes on the image.
        """
        img_result = self.orig_img.copy()

        mask_layer = self.orig_img.copy()
        shapes_found = False

        for pred in self.object_prediction_list:
            color = self._get_color(pred.category.id)

            if pred.mask:
                raw_mask = pred.mask.bool_mask
                mask_bool = np.array(raw_mask).astype(bool)

                if mask_bool.shape[:2] == mask_layer.shape[:2]:
                    shapes_found = True
                    mask_layer[mask_bool] = color

        if shapes_found:  # Add alpha blending only if there are masks
            cv2.addWeighted(
                mask_layer, alpha, img_result, 1 - alpha, 0, img_result
            )

        # Draw boxes and labels on top of masks
        if boxes or labels or conf:
            for pred in self.object_prediction_list:
                bbox = pred.bbox.to_xyxy()
                x1, y1, x2, y2 = map(int, bbox)
                color = self._get_color(pred.category.id)

                if boxes:
                    cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)

                if labels or conf:
                    label_text = ""
                    if labels:
                        label_text += f"{pred.category.name}"
                    if conf:
                        score = pred.score.value
                        label_text += f" {score:.2f}"

                    if label_text:
                        t_size = cv2.getTextSize(
                            label_text, 0, fontScale=0.5, thickness=1
                        )[0]
                        c2 = x1 + t_size[0], y1 - t_size[1] - 3
                        cv2.rectangle(
                            img_result, (x1, y1), c2, color, -1, cv2.LINE_AA
                        )
                        cv2.putText(
                            img_result,
                            label_text,
                            (x1, y1 - 2),
                            0,
                            0.5,
                            [255, 255, 255],
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

        return img_result
