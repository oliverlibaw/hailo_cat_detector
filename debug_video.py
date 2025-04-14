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

# Configuration
MODEL_NAME = "yolo11s_silu_coco--640x640_quant_hailort_hailo8l_1"  # Updated model name
MODEL_ZOO_PATH = "/home/pi5/degirum_model_zoo"  # Updated model zoo path
DETECTION_THRESHOLD = 0.5  # Confidence threshold for detections
VIDEO_PATH = "/home/pi5/Projects/hailo_cat_detector/test_videos/pi_camera_test_640x640.mp4"
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


def load_model():
    """Load the YOLO11s model for detection."""
    try:
        print(f"Loading YOLO11s model: {MODEL_NAME}")
        import degirum as dg
        
        # Check if model zoo path exists
        if not os.path.exists(MODEL_ZOO_PATH):
            print(f"ERROR: Model zoo path not found: {MODEL_ZOO_PATH}")
            print("Please make sure the ModelZoo directory exists and contains the model files.")
            return None
        
        # List available models for debugging
        try:
            available_models = [f for f in os.listdir(MODEL_ZOO_PATH) 
                              if f.endswith(".hef") or os.path.isdir(os.path.join(MODEL_ZOO_PATH, f))]
            print(f"Available models in {MODEL_ZOO_PATH}:")
            for model in available_models:
                print(f"  - {model}")
        except Exception as e:
            print(f"Warning: Could not list contents of model zoo directory: {e}")
        
        # Load the model with optimized parameters
        model = dg.load_model(
            model_name=MODEL_NAME,
            inference_host_address="@local",
            zoo_url=MODEL_ZOO_PATH,
            output_confidence_threshold=DETECTION_THRESHOLD,
            overlay_line_width=3,
            overlay_font_scale=1.5,
            overlay_show_probabilities=True,
            batch_size=1
        )
        
        print(f"Model loaded successfully with confidence threshold: {DETECTION_THRESHOLD}")
        return model
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_frame(frame):
    """Process a single frame for detection."""
    # Convert frame to RGB for better color handling
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model.predict_batch([frame_rgb])
    
    # Process detections
    for result in results:
        for detection in result.results:
            if detection.score >= DETECTION_THRESHOLD:
                # Draw bounding box
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add label
                label = f"{detection.class_name}: {detection.score:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
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
            
            # Process frame
            processed_frame = process_frame(frame)
            
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
    global DETECTION_THRESHOLD  # Move global declaration to beginning of function
    
    parser = argparse.ArgumentParser(description="Process a video file to detect cats.")
    parser.add_argument("--input", "-i", type=str, default="/home/pi5/Projects/hailo_cat_detector/test_videos/pi_camera_test_640x640.mp4",
                        help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, default="debug_output.mp4",
                        help="Path to output video file")
    parser.add_argument("--threshold", "-t", type=float, default=DETECTION_THRESHOLD,
                        help=f"Detection threshold (default: {DETECTION_THRESHOLD})")
    
    args = parser.parse_args()
    
    # Update threshold if provided
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
