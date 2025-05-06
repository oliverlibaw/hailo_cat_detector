#!/usr/bin/env python3
"""
debug_squirrel_video.py
Debug script to test squirrel detection model on a video file using Degirum.
Applies letterboxing for preprocessing and scales bounding boxes back to the
original frame for accurate visualization.
"""

import os
import cv2
import numpy as np
import degirum as dg
from pprint import pprint

# Configuration
MODEL_NAME = "yolov8n_squirrel--640x640_quant_hailort_hailo8l_1"
MODEL_ZOO_PATH = "/home/pi5/degirum_model_zoo"
DETECTION_THRESHOLD = 0.3
INPUT_VIDEO_PATH = "/home/pi5/Downloads/squirrel-test.mp4"
OUTPUT_VIDEO_PATH = "debug_squirrel_output.mp4"

# Squirrel class ID (should match your model)
SQUIRREL_CLASS_ID = 0
COCO_CLASSES = {0: "squirrel"}

# Letterbox utility

def resize_with_letterbox(image, target_shape, padding_value=(0, 0, 0)):
    """
    Resizes an image with letterboxing to fit the target size, preserving aspect ratio.
    Returns the letterboxed image, scale, pad_top, pad_left.
    """
    h, w, c = image.shape
    target_height, target_width = target_shape[1], target_shape[2]
    scale_x = target_width / w
    scale_y = target_height / h
    scale = min(scale_x, scale_y)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    letterboxed_image = np.full((target_height, target_width, c), padding_value, dtype=np.uint8)
    pad_top = (target_height - new_h) // 2
    pad_left = (target_width - new_w) // 2
    letterboxed_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_image
    final_image = np.expand_dims(letterboxed_image, axis=0)
    return final_image, scale, pad_top, pad_left

def reverse_rescale_bboxes(annotations, scale, pad_top, pad_left, original_shape):
    """
    Reverse rescales bounding boxes from the letterbox image to the original image.
    """
    orig_h, orig_w = original_shape
    rescaled_annotations = []
    for ann in annotations:
        bbox = ann['bbox']
        x1, y1, x2, y2 = bbox
        x1 -= pad_left
        y1 -= pad_top
        x2 -= pad_left
        y2 -= pad_top
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        new_ann = ann.copy()
        new_ann['bbox'] = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        rescaled_annotations.append(new_ann)
    return rescaled_annotations

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        label = det['label']
        color = (0, 255, 0)  # Green for squirrel
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1
        text_y = y1 - 5 if y1 > 20 else y1 + 20
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), color, -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def main():
    print(f"Loading model: {MODEL_NAME}")
    model = dg.load_model(
        model_name=MODEL_NAME,
        inference_host_address="@local",
        zoo_url=MODEL_ZOO_PATH,
        output_confidence_threshold=DETECTION_THRESHOLD
    )
    print("Model loaded.")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Failed to open video: {INPUT_VIDEO_PATH}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_count = 0
    total_detections = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Letterbox preprocess
        image_array, scale, pad_top, pad_left = resize_with_letterbox(frame, (1, 640, 640, 3))
        # Run inference
        result = model(image_array)
        detections = []
        if hasattr(result, 'results'):
            for det in result.results:
                if det['score'] >= DETECTION_THRESHOLD and det['category_id'] == SQUIRREL_CLASS_ID:
                    bbox = det['bbox']
                    if isinstance(bbox, dict):
                        x1, y1, x2, y2 = bbox['left'], bbox['top'], bbox['right'], bbox['bottom']
                    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                    else:
                        continue
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'score': det['score'],
                        'label': COCO_CLASSES.get(det['category_id'], f"Class {det['category_id']}")
                    })
        # Reverse letterbox
        detections = reverse_rescale_bboxes(detections, scale, pad_top, pad_left, (height, width))
        total_detections += len(detections)
        # Draw detections
        frame = draw_detections(frame, detections)
        out.write(frame)
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, detections in last frame: {len(detections)}")
    cap.release()
    out.release()
    print(f"Done. Processed {frame_count} frames. Total detections: {total_detections}")
    print(f"Output saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main() 