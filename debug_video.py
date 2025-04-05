#!/usr/bin/env python3
"""
Debug script to test cat detection model on a video file.
This script processes a video file using the same model and parameters as the main.py file,
but focuses only on detection without relay control. It creates an annotated video with
bounding boxes, labels, and confidence scores to evaluate detection accuracy.
"""

import os
import sys
import cv2
import time
import numpy as np
import argparse

# Configuration - matching the main script
MODEL_ZOO_PATH = "/home/pi5/degirum_model_zoo"
MODEL_NAME = "yolo11s_silu_coco--640x640_quant_hailort_hailo8l_1"
DETECTION_THRESHOLD = 0.25  # Slightly lower than main.py for testing
MODEL_INPUT_SIZE = (640, 640)
CAT_CLASS_ID = 15  # COCO dataset cat class ID

# COCO class names for reference
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
    78: 'hair drier', 79: 'toothbrush'
}

# Colors for visualization
COLORS = [
    (0, 255, 0),   # Green
    (255, 0, 0),   # Blue
    (0, 0, 255),   # Red
    (255, 255, 0), # Cyan
    (255, 0, 255), # Magenta
    (0, 255, 255)  # Yellow
]


def load_model():
    """Load the YOLO11s model for detection."""
    try:
        print(f"Loading YOLO11s model: {MODEL_NAME}")
        model_to_load = MODEL_NAME
        zoo_path = MODEL_ZOO_PATH
        
        import degirum as dg
        
        # Check if model zoo path exists
        if not os.path.exists(zoo_path):
            print(f"ERROR: Model zoo path not found: {zoo_path}")
            return None
            
        print(f"Model zoo path verified: {zoo_path}")
        
        # Check if the specific model path exists
        specific_model_path = os.path.join(zoo_path, model_to_load)
        if os.path.exists(specific_model_path):
            print(f"Found model at: {specific_model_path}")
        else:
            print(f"WARNING: Specific model path not found: {specific_model_path}")
            # List available models
            try:
                available_models = [f for f in os.listdir(zoo_path) if f.endswith(".hef") or os.path.isdir(os.path.join(zoo_path, f))]
                print(f"Available models in {zoo_path}:")
                for model in available_models:
                    print(f"  - {model}")
            except Exception as e:
                print(f"Warning: Could not list contents of model zoo directory: {e}")
        
        # Load the model with optimized parameters
        model = dg.load_model(
            model_name=model_to_load,
            inference_host_address="@local",
            zoo_url=zoo_path,
            output_confidence_threshold=DETECTION_THRESHOLD,
            overlay_line_width=3,  # Thicker lines for better visibility
            overlay_font_scale=1.5,  # Font scale for overlay
            overlay_show_probabilities=True,  # Show confidence scores
            output_format="NHWC",  # Use NHWC format for better compatibility
            output_class_activation_threshold=DETECTION_THRESHOLD,  # Match confidence threshold
            batch_size=1  # Set batch size to 1 for consistent performance
        )
        
        print(f"Model loaded successfully with confidence threshold: {DETECTION_THRESHOLD}")
        return model
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_detections(frame, results):
    """Process detection results and extract detections for cats."""
    detections = []
    
    try:
        # YOLO11s model sometimes returns results directly in 'results' and sometimes in 'results.results'
        if hasattr(results, 'results') and results.results:
            result_list = results.results
            print(f"Processing {len(result_list)} detections from results.results")
        elif hasattr(results, 'results') and isinstance(results.results, list):
            result_list = results.results
            print(f"Processing {len(result_list)} detections from results.results list")
        elif isinstance(results, list):
            result_list = results
            print(f"Processing {len(result_list)} detections from direct results list")
        else:
            # No clear list of results found
            result_list = []
            if hasattr(results, '__dict__'):
                print(f"Results attributes: {list(results.__dict__.keys())}")
        
        # Process detections from the result list
        for detection in result_list:
            try:
                # Try multiple approaches to extract bounding box information
                x1, y1, x2, y2, score, class_id = None, None, None, None, None, None
                
                # Method 1: Direct dictionary access
                if isinstance(detection, dict):
                    if 'bbox' in detection:
                        bbox = detection['bbox']
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            x1, y1, x2, y2 = map(int, bbox)
                        
                        score = float(detection.get('score', detection.get('confidence', 0.0)))
                        
                        # Get class ID depending on the model format
                        class_id = int(detection.get('category_id', detection.get('class_id', 0)))
                
                # Method 2: Object attribute access
                elif hasattr(detection, 'bbox'):
                    bbox = detection.bbox
                    
                    # Handle different bbox formats
                    if hasattr(bbox, 'x1') and hasattr(bbox, 'y1') and hasattr(bbox, 'x2') and hasattr(bbox, 'y2'):
                        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                    elif hasattr(bbox, '__getitem__') and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                    
                    # Get score and class ID
                    if hasattr(detection, 'score'):
                        score = float(detection.score)
                    elif hasattr(detection, 'confidence'):
                        score = float(detection.confidence)
                    
                    if hasattr(detection, 'class_id'):
                        class_id = int(detection.class_id)
                    elif hasattr(detection, 'category_id'):
                        class_id = int(detection.category_id)
                
                # Skip invalid or incomplete detections
                if None in (x1, y1, x2, y2, score, class_id):
                    continue
                
                # For this debug script, we want to see all detections, not just cats
                # But highlight cats specifically
                is_cat = (class_id == CAT_CLASS_ID)
                
                # Add detection if it meets threshold
                if score >= DETECTION_THRESHOLD:
                    # Make sure coordinates are within image bounds
                    height, width = frame.shape[:2]
                    x1 = max(0, min(width-1, x1))
                    y1 = max(0, min(height-1, y1))
                    x2 = max(0, min(width-1, x2))
                    y2 = max(0, min(height-1, y2))
                    
                    # Only add if the box has reasonable size
                    if x2 > x1 and y2 > y1 and (x2-x1)*(y2-y1) > 100:  # Minimum area of 100 pixels
                        detections.append((x1, y1, x2, y2, score, class_id, is_cat))
                        print(f"Added detection: bbox={x1},{y1},{x2},{y2}, score={score:.2f}, class={class_id} ({COCO_CLASSES.get(class_id, 'unknown')})")
            
            except Exception as e:
                print(f"Error processing detection: {e}")
        
        # If we have an overlay but no detections, store the overlay for debugging
        if len(detections) == 0 and hasattr(results, 'image_overlay') and results.image_overlay is not None:
            print("No detections found above threshold, but overlay is available")
    
    except Exception as e:
        print(f"Error processing detection results: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Returning {len(detections)} processed detections")
    return detections


def draw_detections(frame, detections):
    """Draw detection bounding boxes and labels on the frame."""
    annotated_frame = frame.copy()
    
    # Add header with model info
    height, width = annotated_frame.shape[:2]
    model_text = f"Model: YOLO11s"
    threshold_text = f"Threshold: {DETECTION_THRESHOLD:.2f}"
    
    cv2.rectangle(annotated_frame, (0, 0), (300, 70), (0, 0, 0), -1)
    cv2.putText(annotated_frame, model_text, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, threshold_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw each detection
    for detection in detections:
        x1, y1, x2, y2, score, class_id, is_cat = detection
        
        # Get class name
        class_name = COCO_CLASSES.get(class_id, f"Unknown class {class_id}")
        
        # Get color - use special color for cats
        if is_cat:
            color = (0, 255, 0)  # Green for cats
            thickness = 3        # Thicker lines for cats
        else:
            color = COLORS[class_id % len(COLORS)]
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Add label with confidence
        label = f"{class_name}: {score:.2f}"
        
        # Calculate text size to create better background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw filled rectangle for text background
        cv2.rectangle(annotated_frame, (x1, y1-30), (x1+text_width+10, y1), color, -1)
        
        # Draw text
        cv2.putText(annotated_frame, label, (x1+5, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        # For cats, add additional highlight
        if is_cat:
            # Draw crosshair at center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.drawMarker(annotated_frame, (center_x, center_y), 
                          (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, timestamp, (width - 250, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add frame counter in bottom right
    if hasattr(draw_detections, 'frame_count'):
        draw_detections.frame_count += 1
    else:
        draw_detections.frame_count = 1
    
    cv2.putText(annotated_frame, f"Frame: {draw_detections.frame_count}", (width - 200, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated_frame


def process_video(input_path, output_path):
    """Process a video file, detecting cats and creating an annotated output video."""
    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Load the model
    model = load_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return False
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' or other codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    processed_frames = 0
    start_time = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for inference if needed
            if frame.shape[:2] != MODEL_INPUT_SIZE:
                inference_frame = cv2.resize(frame, MODEL_INPUT_SIZE)
            else:
                inference_frame = frame.copy()
            
            # Run inference
            results_generator = model.predict_batch([inference_frame])
            results = next(results_generator)
            
            # Process detections
            detections = process_detections(frame, results)
            
            # Draw detections on the original frame
            annotated_frame = draw_detections(frame, detections)
            
            # Write the annotated frame
            out.write(annotated_frame)
            
            # Display progress
            processed_frames += 1
            if processed_frames % 10 == 0:
                elapsed = time.time() - start_time
                percent_complete = (processed_frames / frame_count) * 100
                remaining = (elapsed / processed_frames) * (frame_count - processed_frames)
                print(f"Processed {processed_frames}/{frame_count} frames ({percent_complete:.1f}%) - ETA: {remaining:.1f}s")
            
            # Save a sample frame every 100 frames
            if processed_frames % 100 == 0:
                sample_path = f"sample_frame_{processed_frames}.jpg"
                cv2.imwrite(sample_path, annotated_frame)
                print(f"Saved sample frame to {sample_path}")
                
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        cap.release()
        out.release()
        print(f"Processed {processed_frames} frames in {time.time() - start_time:.1f} seconds")
        print(f"Output video saved to {output_path}")
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process a video file to detect cats.")
    parser.add_argument("--input", "-i", type=str, default="/home/pi5/Downloads/cat_test1.MOV",
                        help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, default="debug_output.mp4",
                        help="Path to output video file")
    parser.add_argument("--threshold", "-t", type=float, default=DETECTION_THRESHOLD,
                        help=f"Detection threshold (default: {DETECTION_THRESHOLD})")
    
    args = parser.parse_args()
    
    # Update threshold if provided
    global DETECTION_THRESHOLD
    if args.threshold != DETECTION_THRESHOLD:
        DETECTION_THRESHOLD = args.threshold
        print(f"Using custom detection threshold: {DETECTION_THRESHOLD}")
    
    # Process the video
    success = process_video(args.input, args.output)
    
    if success:
        print("Video processing completed successfully!")
    else:
        print("Video processing failed.")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
