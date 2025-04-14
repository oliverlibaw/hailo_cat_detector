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
import degirum as dg

# Configuration
MODEL_NAME = "yolo11s_silu_coco--640x640_quant_hailort_hailo8l_1"
MODEL_ZOO_PATH = "/home/pi5/degirum_model_zoo"
DETECTION_THRESHOLD = 0.5
OUTPUT_PATH = "debug_output.mp4"

# COCO class IDs for cats and dogs
CAT_CLASS_ID = 15  # cat
DOG_CLASS_ID = 16  # dog

# COCO class names (only including cats and dogs for clarity)
COCO_CLASSES = {
    CAT_CLASS_ID: "cat",
    DOG_CLASS_ID: "dog"
}

# Colors for visualization (using distinct colors for cats and dogs)
COLORS = [
    (0, 255, 0),    # Green for cats
    (255, 0, 0)     # Red for dogs
]


def get_video_path():
    """Prompt user for video file path."""
    while True:
        video_path = input("Enter the path to the video file: ")
        if os.path.exists(video_path):
            return video_path
        print(f"Error: File not found: {video_path}")
        print("Please enter a valid video file path.")


def load_model():
    """Load the YOLO11s model for detection."""
    try:
        print(f"Loading YOLO11s model: {MODEL_NAME}")
        
        # Check if model zoo path exists
        if not os.path.exists(MODEL_ZOO_PATH):
            print(f"ERROR: Model zoo path not found: {MODEL_ZOO_PATH}")
            print("Please make sure the ModelZoo directory exists and contains the model files.")
            return None
        
        # Load the model with minimal parameters
        model = dg.load_model(
            model_name=MODEL_NAME,
            inference_host_address="@local",
            zoo_url=MODEL_ZOO_PATH
        )
        
        print(f"Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_frame(frame, model):
    """Process a single frame for detection."""
    # Convert frame to RGB for better color handling
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model.predict_batch([frame_rgb])
    
    # Process detections
    for result in results:
        for detection in result.results:
            try:
                # Access detection properties
                score = detection['score']
                category_id = detection['category_id']
                label = detection['label']
                
                # Print all detection info for debugging
                print(f"Detection: {label} (ID: {category_id}) with confidence {score:.2f}")
                
                # Only process cats and dogs
                if category_id not in [CAT_CLASS_ID, DOG_CLASS_ID]:
                    continue
                
                if score >= DETECTION_THRESHOLD:
                    # Get bounding box coordinates
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Get color based on class
                    color = COLORS[0] if category_id == CAT_CLASS_ID else COLORS[1]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Add label with confidence score
                    label_text = f"{label}: {score:.2f}"
                    cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Print detection info for debugging
                    print(f"Detected {label} with confidence {score:.2f} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                    
            except KeyError as e:
                print(f"Warning: Missing key in detection: {e}")
                print("Detection structure:", detection)
                continue
    
    return frame


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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    processed_frames = 0
    start_time = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with the model
            processed_frame = process_frame(frame, model)
            
            # Write frame
            out.write(processed_frame)
            
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
                cv2.imwrite(sample_path, processed_frame)
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
    global DETECTION_THRESHOLD
    
    # Get video path from user
    video_path = get_video_path()
    
    parser = argparse.ArgumentParser(description="Process a video file to detect cats and dogs.")
    parser.add_argument("--output", "-o", type=str, default=OUTPUT_PATH,
                        help="Path to output video file")
    parser.add_argument("--threshold", "-t", type=float, default=DETECTION_THRESHOLD,
                        help=f"Detection threshold (default: {DETECTION_THRESHOLD})")
    
    args = parser.parse_args()
    
    # Update threshold if provided
    if args.threshold != DETECTION_THRESHOLD:
        DETECTION_THRESHOLD = args.threshold
        print(f"Using custom detection threshold: {DETECTION_THRESHOLD}")
    
    # Process the video
    success = process_video(video_path, args.output)
    
    if success:
        print("Video processing completed successfully!")
    else:
        print("Video processing failed.")
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
