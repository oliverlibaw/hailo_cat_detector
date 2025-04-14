#!/usr/bin/env python3
"""
Debug script to test cat/dog detection model on a video file using Degirum.
Applies letterboxing for preprocessing and scales bounding boxes back to the
original frame for accurate visualization.
"""

import os
import sys
import cv2
import time
import numpy as np
import argparse
import degirum as dg
from pprint import pprint # For potentially debugging raw results

# Configuration
# Ensure your model name is correct
MODEL_NAME = "yolo11s_silu_coco--640x640_quant_hailort_hailo8l_1"
# MODEL_NAME = "yolov5s_coco--640x640_quant_hailort_hailo8l_1" # Example if using YOLOv5
MODEL_ZOO_PATH = "/home/pi5/degirum_model_zoo" # Ensure this path is correct
DEFAULT_THRESHOLD = 0.3 # Default detection threshold
OUTPUT_PATH = "debug_output.mp4" # Default output file

# COCO class IDs for cats and dogs (Verify these match your model's output)
# Common COCO IDs: Cat=16, Dog=17 in some versions (0-indexed)
# Or Cat=15, Dog=16 if dataset has background class or is 1-indexed
# Double-check your model's specific label mapping if issues persist.
CAT_CLASS_ID = 15 # Assuming 1-based index or specific mapping
DOG_CLASS_ID = 16 # Assuming 1-based index or specific mapping
TARGET_CLASSES = {
    CAT_CLASS_ID: "cat",
    DOG_CLASS_ID: "dog"
}
TARGET_CLASS_IDS = set(TARGET_CLASSES.keys())

# Colors for visualization (using distinct colors for cats and dogs)
COLORS = {
    CAT_CLASS_ID: (0, 255, 0),    # Green for cats
    DOG_CLASS_ID: (255, 0, 0)     # Blue for dogs (Changed from Red for better distinction if needed)
}
DEFAULT_COLOR = (255, 255, 255) # White for other classes if shown

# --- Helper Functions based on Degirum Examples ---

def resize_with_letterbox(image, target_shape=(640, 640), padding_value=(114, 114, 114)):
    """
    Resizes an image (NumPy array) with letterboxing to fit the target size, preserving aspect ratio.
    Adapted from Degirum example to work with frames instead of image paths.

    Parameters:
        image (ndarray): Input image (frame) in RGB format.
        target_shape (tuple): Target (height, width).
        padding_value (tuple): RGB values for padding.

    Returns:
        letterboxed_image (ndarray): The resized image with letterboxing (H, W, C).
        scale (float): Scaling ratio applied to the original image.
        pad_top (int): Padding applied to the top.
        pad_left (int): Padding applied to the left.
    """
    h, w = image.shape[:2]
    target_height, target_width = target_shape

    # Calculate the scaling factor
    scale = min(target_width / w, target_height / h)

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create letterbox background
    letterboxed_image = np.full((target_height, target_width, 3), padding_value, dtype=np.uint8)

    # Calculate padding
    pad_top = (target_height - new_h) // 2
    pad_left = (target_width - new_w) // 2

    # Place the resized image onto the letterbox background
    letterboxed_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_image

    return letterboxed_image, scale, pad_top, pad_left


def reverse_rescale_bboxes(annotations, scale, pad_top, pad_left, original_shape):
    """
    Reverse rescales bounding boxes from the letterbox image to the original image.
    Adapted from Degirum example.

    Parameters:
        annotations (list of dicts): Detections from the model, must contain 'bbox'.
        scale (float): Scale factor used in letterboxing.
        pad_top (int): Top padding used in letterboxing.
        pad_left (int): Left padding used in letterboxing.
        original_shape (tuple): Original frame shape (height, width).

    Returns:
        rescaled_annotations (list of dicts): Annotations with 'bbox' scaled to original frame.
    """
    orig_h, orig_w = original_shape
    rescaled_annotations = []

    for ann in annotations:
        bbox = ann['bbox']
        x1, y1, x2, y2 = bbox

        # Reverse padding
        x1 -= pad_left
        y1 -= pad_top
        x2 -= pad_left
        y2 -= pad_top

        # Reverse scaling
        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale

        # Clip to original dimensions
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        new_ann = ann.copy()
        # Ensure bbox coordinates are integers for drawing
        new_ann['bbox'] = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        rescaled_annotations.append(new_ann)

    return rescaled_annotations

# --- Main Script Functions ---

def get_video_path():
    """Prompt user for video file path."""
    while True:
        video_path = input("Enter the path to the video file: ")
        if os.path.exists(video_path):
            return video_path
        print(f"Error: File not found: {video_path}")
        print("Please enter a valid video file path.")

def preprocess_frame(frame, target_shape=(640, 640)):
    """
    Preprocess frame for model input according to DeGirum requirements.
    
    Parameters:
        frame (ndarray): Input frame in BGR format
        target_shape (tuple): Target (height, width)
        
    Returns:
        preprocessed_frame (ndarray): Preprocessed frame ready for model input
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to target shape using bilinear interpolation
    resized = cv2.resize(frame_rgb, target_shape, interpolation=cv2.INTER_LINEAR)
    
    # Ensure the image is in uint8 format (0-255 range)
    normalized = resized.astype(np.uint8)
    
    # Add batch dimension and ensure correct shape
    batched = np.expand_dims(normalized, axis=0)
    
    # Ensure the shape matches exactly what the model expects
    if batched.shape != (1, 640, 640, 3):
        raise ValueError(f"Invalid shape: {batched.shape}. Expected (1, 640, 640, 3)")
    
    return batched

def load_model(model_name, zoo_path):
    """Load the specified Degirum model."""
    try:
        print(f"Loading model: {model_name} from {zoo_path}")
        if not os.path.exists(zoo_path):
            print(f"ERROR: Model zoo path not found: {zoo_path}")
            return None

        # Load the model with specific backend settings
        model = dg.load_model(
            model_name=model_name,
            inference_host_address="@local",
            zoo_url=zoo_path,
            image_backend="opencv"
        )

        # Print model information
        print("\nModel Information:")
        print(f"Model name: {model_name}")
        print(f"Model zoo path: {zoo_path}")
        
        # Get model input shape and format
        input_shape = model.input_shape[0]  # Get first input shape (batch, height, width, channels)
        print(f"Model input shape: {input_shape}")
        
        # Run a test inference to verify model works
        print("\nRunning test inference...")
        test_frame = np.zeros((input_shape[1], input_shape[2], input_shape[3]), dtype=np.uint8)
        test_batch = preprocess_frame(test_frame)  # Use the preprocessing function
        
        # Run inference and collect results
        results = list(model.predict_batch([test_batch]))  # Convert generator to list
        
        if results and results[0].results:
            print("\nSample detection structure:")
            print(results[0].results[0])
            print()
            
            # Print available classes from the first result
            seen_classes = set()
            for detection in results[0].results:
                if 'label' in detection:
                    seen_classes.add(detection['label'])
            
            print("Available classes in detections:")
            for class_name in sorted(seen_classes):
                print(f"- {class_name}")
            print()
        else:
            print("Warning: No detections in test inference")
        
        print("Model loaded successfully.")
        return model
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return None

def process_frame(frame, model, detection_threshold, show_all=False):
    """Process a single frame through the model."""
    try:
        # Preprocess the frame
        preprocessed = preprocess_frame(frame)
        
        # Run inference
        results = list(model.predict_batch([preprocessed]))[0]
        
        # Process detections
        detections = []
        for detection in results.results:
            if 'bbox' in detection and 'score' in detection:
                if detection['score'] >= detection_threshold:
                    if show_all or detection.get('category_id') in TARGET_CLASS_IDS:
                        detections.append(detection)
        
        return detections
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return []

def process_video(input_path, output_path, model, threshold, show_all):
    """Process a video file frame by frame."""
    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Using detection threshold: {threshold}")
    if show_all:
        print("Showing detections for ALL classes above threshold.")
    else:
        print(f"Showing detections only for target classes: {list(TARGET_CLASSES.values())}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0: # Frame count might not be available for all streams/formats
        print("Warning: Frame count not available.")
        frame_count = float('inf') # Avoid division by zero

    print(f"Input Video: {width}x{height}, {fps:.2f} FPS, Frame count ~{frame_count}")

    # Output writer using original dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    processed_frames = 0
    start_time = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            # Process the frame
            processed_frame = process_frame(frame, model, threshold, show_all)

            # Write the processed frame (with boxes drawn)
            out.write(processed_frame)

            processed_frames += 1
            if processed_frames % 30 == 0: # Print progress every ~second
                elapsed = time.time() - start_time
                if frame_count != float('inf'):
                    percent_complete = (processed_frames / frame_count) * 100
                    if processed_frames > 0:
                         eta = (elapsed / processed_frames) * (frame_count - processed_frames)
                         print(f"Processed {processed_frames}/{frame_count} frames ({percent_complete:.1f}%) - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
                    else:
                         print(f"Processed {processed_frames}/{frame_count} frames ({percent_complete:.1f}%) - Elapsed: {elapsed:.1f}s")
                else:
                    print(f"Processed {processed_frames} frames - Elapsed: {elapsed:.1f}s")

            # Optional: Save sample frames periodically
            # if processed_frames % 100 == 0:
            #     sample_path = f"sample_frame_{processed_frames}.jpg"
            #     cv2.imwrite(sample_path, processed_frame)
            #     print(f"Saved sample frame to {sample_path}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nError processing video: {e}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed_total = time.time() - start_time
        print(f"\nFinished processing.")
        print(f"Processed {processed_frames} frames in {elapsed_total:.1f} seconds.")
        if processed_frames > 0:
            avg_fps = processed_frames / elapsed_total
            print(f"Average processing speed: {avg_fps:.2f} FPS")
        cap.release()
        out.release()
        cv2.destroyAllWindows() # Close any OpenCV windows if used for display
        print(f"Output video saved to {output_path}")

    return True

def main():
    """Main function to parse arguments and start video processing."""
    parser = argparse.ArgumentParser(description="Process video for cat/dog detection using Degirum.")
    parser.add_argument("video_path", nargs='?', default=None,
                        help="Path to the input video file (optional, will prompt if not provided)")
    parser.add_argument("--model", "-m", type=str, default=MODEL_NAME,
                        help=f"Name of the Degirum model (default: {MODEL_NAME})")
    parser.add_argument("--zoo", "-z", type=str, default=MODEL_ZOO_PATH,
                        help=f"Path to the Degirum model zoo (default: {MODEL_ZOO_PATH})")
    parser.add_argument("--output", "-o", type=str, default=OUTPUT_PATH,
                        help=f"Path to output annotated video file (default: {OUTPUT_PATH})")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Detection confidence threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--show-all", "-a", action="store_true",
                        help="Show detections for all classes, not just cats/dogs.")
    # Retain --debug flag if needed for other debug prints, otherwise -a covers showing more boxes
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable extra debug prints (currently minimal use).")

    args = parser.parse_args()

    # Get video path if not provided via argument
    input_video_path = args.video_path if args.video_path else get_video_path()

    # Load the model
    model = load_model(args.model, args.zoo)
    if model is None:
        print("Exiting due to model loading failure.")
        return 1

    # Process the video
    success = process_video(input_video_path, args.output, model, args.threshold, args.show_all)

    if success:
        print("Video processing completed successfully!")
    else:
        print("Video processing failed.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())