#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from robot_msgs.msg import VisionMsg
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import nms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import easyocr
import Levenshtein
import os
import time


class RPN(nn.Module):
    def __init__(self, in_channels_list, num_anchors):
        super(RPN, self).__init__()
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.ModuleList()
        self.cls_score = nn.ModuleList()
        self.bbox_pred = nn.ModuleList()
        for in_channels in in_channels_list:
            self.conv.append(nn.Conv2d(in_channels, 256, 3, 1, 1))
            self.bn.append(nn.BatchNorm2d(256))
            self.dropout.append(nn.Dropout(0.5))
            self.cls_score.append(nn.Conv2d(256, num_anchors, 1, 1))
            self.bbox_pred.append(nn.Conv2d(256, num_anchors * 4, 1, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        cls_scores, bbox_preds = [], []
        for i, x in enumerate(features):
            t = self.relu(self.bn[i](self.conv[i](x)))
            t = self.dropout[i](t)
            cls_scores.append(self.cls_score[i](t))
            bbox_preds.append(self.bbox_pred[i](t))
        return cls_scores, bbox_preds


class OCRInferenceNode(Node):
    def __init__(self):
        super().__init__("ocr_inference_node")

        self.image_subscription = self.create_subscription(
            Image, "/ocr_request", self.image_callback, 10
        )
        self.vision_publisher = self.create_publisher(VisionMsg, "turtle_vision", 10)

        self.bridge = CvBridge()
        self.setup_models()

        self.db_strings = [
            "초코프렌즈우유",
            "계란과자",
            "정우경",
            "홍지현",
            "음료수",
            "과자",
        ]

        self.get_logger().info("OCR 추론 노드가 시작되었습니다.")

    def setup_models(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.get_logger().info(f"디바이스: {self.device}")

            self.reader = easyocr.Reader(["ko"], gpu=torch.cuda.is_available())

            self.backbone = (
                resnet_fpn_backbone(
                    backbone_name="resnet50", weights="DEFAULT", trainable_layers=3
                )
                .to(self.device)
                .eval()
            )

            self.rpn = RPN([256] * 5, 3).to(self.device).eval()

            model_paths = [
                os.path.join(
                    os.path.dirname(__file__), "..", "models", "final_model.pth"
                ),
                os.path.join(
                    os.path.expanduser("~"),
                    "turtlebot3_ws",
                    "src",
                    "turtlebot_robot",
                    "robot_vision",
                    "models",
                    "final_model.pth",
                ),
                "/opt/ros/humble/share/robot_vision/models/final_model.pth",
            ]

            model_loaded = False
            for model_path in model_paths:
                try:
                    if os.path.exists(model_path):
                        checkpoint = torch.load(
                            model_path, map_location=self.device, weights_only=True
                        )

                        if (
                            "model_state_dict" in checkpoint
                            and "backbone_state_dict" in checkpoint
                        ):
                            self.rpn.load_state_dict(checkpoint["model_state_dict"])
                            self.backbone.load_state_dict(
                                checkpoint["backbone_state_dict"]
                            )
                            self.get_logger().info(
                                f"모델 가중치를 성공적으로 로드했습니다: {model_path}"
                            )
                            model_loaded = True
                            break
                except Exception as e:
                    try:
                        checkpoint = torch.load(
                            model_path, map_location=self.device, weights_only=False
                        )
                        if (
                            "model_state_dict" in checkpoint
                            and "backbone_state_dict" in checkpoint
                        ):
                            self.rpn.load_state_dict(checkpoint["model_state_dict"])
                            self.backbone.load_state_dict(
                                checkpoint["backbone_state_dict"]
                            )
                            self.get_logger().info(
                                f"모델 가중치를 성공적으로 로드했습니다 (fallback): {model_path}"
                            )
                            model_loaded = True
                            break
                    except Exception:
                        continue

            if not model_loaded:
                self.get_logger().warn(
                    "모델 체크포인트를 찾을 수 없습니다. 기본 가중치를 사용합니다."
                )

            self.transform = A.Compose(
                [
                    A.Resize(256, 256),
                    A.CenterCrop(224, 224),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

        except Exception as e:
            self.get_logger().error(f"모델 초기화 실패: {str(e)}")

    def image_callback(self, msg):
        try:
            start_time = time.time()

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.perform_ocr_inference(cv_image)

            end_time = time.time()
            processing_time_ms = int((end_time - start_time) * 1000)

            self.publish_results(results, processing_time_ms)

        except Exception as e:
            self.get_logger().error(f"이미지 처리 오류: {str(e)}")

    def perform_ocr_inference(self, cv_image):
        try:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)

            boxes, scores = self.infer_from_frame(pil_image)

            results = []
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = self.expand_box(
                    box, 0.1, cv_image.shape[1], cv_image.shape[0]
                )

                crop = pil_image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                processed = self.preprocess_image(np.array(crop))

                ocr_results = self.reader.readtext(
                    processed,
                    detail=1,
                    decoder="greedy",
                    rotation_info=[0, 90, 180, 270],
                )

                best_match, best_score, best_text = self.find_best_ocr_match(
                    ocr_results
                )

                if best_match:
                    results.append(
                        {
                            "text": best_match,
                            "confidence": best_score,
                            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                            "raw_text": best_text,
                        }
                    )

                    self.get_logger().info(
                        f'[물품 {i+1}] OCR: "{best_text}" → "{best_match}" (신뢰도: {best_score:.2f}) '
                        f"위치: [{int(x_min)}, {int(y_min)}, {int(x_max)}, {int(y_max)}]"
                    )

            return results

        except Exception as e:
            self.get_logger().error(f"OCR 추론 오류: {str(e)}")
            return []

    def find_best_ocr_match(self, ocr_results):
        best_match = None
        best_score = 0
        best_text = ""

        for res in ocr_results:
            if len(res) >= 3:
                _, text, conf = res
                cleaned = "".join(filter(str.isalnum, text))

                match = self.find_best_match(cleaned, self.db_strings)
                if match:
                    sim = Levenshtein.ratio(cleaned, match)
                    if sim > best_score:
                        best_score = sim
                        best_match = match
                        best_text = text

        return best_match, best_score, best_text

    def publish_results(self, results, processing_time_ms):
        vision_msg = VisionMsg()
        vision_msg.fps = processing_time_ms

        if results:
            best_result = max(results, key=lambda x: x["confidence"])
            vision_msg.ocr_detected = True

            bbox_data = f"{best_result['text']}|{best_result['bbox'][0]},{best_result['bbox'][1]},{best_result['bbox'][2]},{best_result['bbox'][3]}|{best_result['confidence']:.2f}"
            vision_msg.ocr_data = bbox_data

            self.get_logger().info(
                f'물품 인식 완료: {best_result["text"]} (신뢰도: {best_result["confidence"]:.2f}) '
                f"처리시간: {processing_time_ms}ms"
            )
        else:
            vision_msg.ocr_detected = False
            vision_msg.ocr_data = ""
            self.get_logger().info(f"물품 인식 실패 - 처리시간: {processing_time_ms}ms")

        self.vision_publisher.publish(vision_msg)

    def generate_anchors(self, feature_maps, sizes, ratios, strides):
        anchors = []
        for fm, size, stride in zip(feature_maps, sizes, strides):
            anchors_per_fm = self.generate_anchors_per_feature_map(
                fm.shape[-2:], size, ratios, stride
            )
            anchors.append(anchors_per_fm)
        return torch.cat(anchors, dim=0)

    def generate_anchors_per_feature_map(
        self, feature_map_size, scales, ratios, stride
    ):
        height, width = feature_map_size
        anchors = []
        for y in range(height):
            for x in range(width):
                cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
                for scale in scales:
                    for ratio in ratios:
                        w = scale * np.sqrt(ratio)
                        h = scale / np.sqrt(ratio)
                        anchors.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        return torch.tensor(anchors, dtype=torch.float32)

    def decode_bbox(self, anchors, bbox_preds):
        centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        sizes = anchors[:, 2:] - anchors[:, :2]
        dx, dy, dw, dh = (
            bbox_preds[:, 0],
            bbox_preds[:, 1],
            bbox_preds[:, 2],
            bbox_preds[:, 3],
        )

        pred_centers = torch.stack(
            [dx * sizes[:, 0] + centers[:, 0], dy * sizes[:, 1] + centers[:, 1]], dim=1
        )
        pred_sizes = torch.stack(
            [torch.exp(dw) * sizes[:, 0], torch.exp(dh) * sizes[:, 1]], dim=1
        )

        return torch.cat(
            [pred_centers - pred_sizes / 2, pred_centers + pred_sizes / 2], dim=1
        )

    def expand_box(self, box, expand_ratio, w, h):
        x_min, y_min, x_max, y_max = box
        bw, bh = x_max - x_min, y_max - y_min
        x_min = max(0, x_min - bw * expand_ratio)
        y_min = max(0, y_min - bh * expand_ratio)
        x_max = min(w, x_max + bw * expand_ratio)
        y_max = min(h, y_max + bh * expand_ratio)
        return [x_min, y_min, x_max, y_max]

    def preprocess_image(self, image):
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0).apply(l)
        lab = cv2.merge((cl, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        image = cv2.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        return image

    def find_best_match(self, text, db_strings, threshold=0.6):
        best_match = None
        max_ratio = 0
        for candidate in db_strings:
            ratio = Levenshtein.ratio(text, candidate)
            if ratio > max_ratio:
                max_ratio = ratio
                best_match = candidate
        return best_match if max_ratio >= threshold else None

    def infer_from_frame(
        self,
        pil_image,
        sizes=[[32], [64], [128], [256], [512]],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64],
        score_threshold=0.7,
        nms_threshold=0.3,
    ):

        orig_w, orig_h = pil_image.size
        transformed = self.transform(image=np.array(pil_image))
        image_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.backbone(image_tensor)
            feature_maps = list(features.values())
            cls_scores, bbox_preds = self.rpn(feature_maps)
            anchors = self.generate_anchors(feature_maps, sizes, ratios, strides).to(
                self.device
            )

            cls_scores_all, bbox_preds_all = [], []
            for cls, bbox in zip(cls_scores, bbox_preds):
                N, A, H, W = cls.shape
                cls = cls.permute(0, 2, 3, 1).reshape(N, -1)
                bbox = bbox.permute(0, 2, 3, 1).reshape(N, -1, 4)
                cls_scores_all.append(cls)
                bbox_preds_all.append(bbox)

            pred_cls = torch.cat(cls_scores_all, dim=1).squeeze(0)
            pred_reg = torch.cat(bbox_preds_all, dim=1).squeeze(0)

            pred_cls = torch.sigmoid(pred_cls)
            pred_boxes = self.decode_bbox(anchors, pred_reg)

            scores = pred_cls.cpu().numpy()
            boxes = pred_boxes.cpu().numpy()

        indices = np.where(scores > score_threshold)[0]
        filtered_boxes = boxes[indices]
        filtered_scores = scores[indices]

        if len(filtered_boxes) > 0:
            boxes_tensor = torch.tensor(filtered_boxes, dtype=torch.float32).to(
                self.device
            )
            scores_tensor = torch.tensor(filtered_scores, dtype=torch.float32).to(
                self.device
            )
            keep_indices = nms(boxes_tensor, scores_tensor, nms_threshold)
            final_boxes = boxes_tensor[keep_indices].cpu().numpy()
            final_scores = scores_tensor[keep_indices].cpu().numpy()

            pad_x, pad_y = (256 - 224) // 2, (256 - 224) // 2
            final_boxes[:, [0, 2]] += pad_x
            final_boxes[:, [1, 3]] += pad_y
            scale_x, scale_y = orig_w / 256, orig_h / 256
            final_boxes[:, [0, 2]] *= scale_x
            final_boxes[:, [1, 3]] *= scale_y

            final_boxes = np.clip(
                final_boxes, 0, [orig_w - 1, orig_h - 1, orig_w - 1, orig_h - 1]
            )
        else:
            final_boxes = np.array([]).reshape(0, 4)
            final_scores = np.array([])

        return final_boxes, final_scores


def main(args=None):
    rclpy.init(args=args)
    node = OCRInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
