#!/usr/bin/env python3
"""
debug_squirrel_video.py
Debug script to test squirrel detection model on a video file using Degirum.
Uses Degirum's built-in preprocessing (no manual letterboxing).
Draws bounding boxes and saves output video for model evaluation.
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
INPUT_VIDEO_PATH = "/home/pi5/Downloads/squirrel_distance_test.mp4"
OUTPUT_VIDEO_PATH = "debug_squirrel_output.mp4"

# Squirrel class ID (should match your model)
SQUIRREL_CLASS_ID = 0
COCO_CLASSES = {0: "squirrel"}

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
        # Convert BGR to RGB for Degirum
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Run inference (Degirum handles resizing/letterboxing internally)
        result = model(rgb_frame)
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
                    # Ensure coordinates are integers and within frame
                    x1 = max(0, min(int(round(x1)), width - 1))
                    y1 = max(0, min(int(round(y1)), height - 1))
                    x2 = max(0, min(int(round(x2)), width - 1))
                    y2 = max(0, min(int(round(y2)), height - 1))
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'score': det['score'],
                        'label': COCO_CLASSES.get(det['category_id'], f"Class {det['category_id']}")
                    })
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