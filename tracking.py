from ultralytics import YOLO
import cv2
import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Load YOLO model
model = YOLO("yolo11n.onnx")

# Open video file
video_path = "traffic_camera.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize video writer
output_path = "output.mp4"
writer = cv2.VideoWriter(output_path,
                        cv2.VideoWriter_fourcc(*'avc1'),
                        fps,
                        (width, height))

# Performance tracking variables
start_time = time.time()
processing_times = []

# Model input size
input_size = 416

# Batch size
batch_size = 8

# Dictionary to store car trails
car_trails = defaultdict(list)
# Maximum number of positions to keep in trail
MAX_TRAIL_LENGTH = 30
# IOU threshold for tracking
IOU_THRESHOLD = 0.3

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def update_trails(detections, frame_number):
    """Update car trails with new detections"""
    current_boxes = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        current_boxes.append((x1, y1, x2, y2))
    
    # Match current detections with existing trails
    matched_trails = set()
    matched_detections = set()
    
    for i, current_box in enumerate(current_boxes):
        best_iou = IOU_THRESHOLD
        best_trail_id = None
        
        for trail_id in car_trails:
            if trail_id in matched_trails:
                continue
                
            if car_trails[trail_id]:
                last_box = car_trails[trail_id][-1][1]
                iou = calculate_iou(current_box, last_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_trail_id = trail_id
        
        if best_trail_id is not None:
            car_trails[best_trail_id].append((frame_number, current_box))
            matched_trails.add(best_trail_id)
            matched_detections.add(i)
    
    # Create new trails for unmatched detections
    for i in range(len(current_boxes)):
        if i not in matched_detections:
            new_trail_id = len(car_trails)
            car_trails[new_trail_id].append((frame_number, current_boxes[i]))
    
    # Prune old positions from trails
    for trail_id in car_trails:
        if len(car_trails[trail_id]) > MAX_TRAIL_LENGTH:
            car_trails[trail_id] = car_trails[trail_id][-MAX_TRAIL_LENGTH:]

def draw_trails(frame, frame_number):
    """Draw trails on the frame with smoothed positions"""
    for trail_id, positions in car_trails.items():
        # Only draw trails that have a recent detection
        if positions and positions[-1][0] >= frame_number - MAX_TRAIL_LENGTH:
            # Collect centers
            centers = []
            for pos in positions:
                box = pos[1]
                center = (
                    (box[0] + box[2]) // 2,
                    (box[1] + box[3]) // 2
                )
                centers.append(center)
            
            # Apply exponential smoothing to centers
            smoothed_centers = []
            alpha_smoothing = 0.2  # Smoothing factor (adjust as needed)
            prev_smoothed_center = centers[0]
            smoothed_centers.append(prev_smoothed_center)
            for i in range(1, len(centers)):
                curr_center = centers[i]
                smoothed_center = (
                    int(alpha_smoothing * curr_center[0] + (1 - alpha_smoothing) * prev_smoothed_center[0]),
                    int(alpha_smoothing * curr_center[1] + (1 - alpha_smoothing) * prev_smoothed_center[1])
                )
                smoothed_centers.append(smoothed_center)
                prev_smoothed_center = smoothed_center
            
            # Draw lines connecting smoothed positions
            for i in range(1, len(smoothed_centers)):
                prev_center = smoothed_centers[i-1]
                curr_center = smoothed_centers[i]
                
                # Calculate alpha based on position age
                alpha = 0.7 * (1 - (frame_number - positions[i][0]) / MAX_TRAIL_LENGTH)
                if alpha > 0:
                    # Draw semi-transparent line
                    overlay = frame.copy()
                    cv2.line(overlay, prev_center, curr_center, (0, 255, 0), 2)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def pad_batch(batch_frames, batch_size):
    pad_size = batch_size - len(batch_frames)
    if pad_size > 0:
        last_frame = batch_frames[-1]
        for _ in range(pad_size):
            batch_frames.append(last_frame)
    return batch_frames

try:
    # Create progress bar
    pbar = tqdm(total=total_frames, desc='Processing video', unit='frames')

    frame_count = 0
    batch_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        batch_frames.append(frame)
        frame_count += 1

        # If batch is full or last frame
        if len(batch_frames) == batch_size or frame_count == total_frames:
            # Pad the batch if needed
            if len(batch_frames) < batch_size:
                original_batch_length = len(batch_frames)
                batch_frames = pad_batch(batch_frames, batch_size)
            else:
                original_batch_length = batch_size

            frame_start_time = time.time()

            # Run inference with explicit image size on the batch
            results = model(batch_frames, imgsz=input_size)

            # Process each frame in the batch
            for i, result in enumerate(results[:original_batch_length]):
                # Filter for car class (typically class 2 in COCO dataset)
                car_detections = result.boxes[result.boxes.cls == 2]
                
                # Update trails with new detections
                update_trails(car_detections, frame_count - original_batch_length + i)
                
                # Draw trails on the frame
                current_frame = batch_frames[i].copy()
                draw_trails(current_frame, frame_count - original_batch_length + i)
                
                # Draw current detections
                result.boxes = car_detections
                annotated_frame = result.plot(img=current_frame)
                
                writer.write(annotated_frame)
                pbar.update(1)

            # Calculate batch processing time
            frame_time = time.time() - frame_start_time
            avg_frame_time = frame_time / original_batch_length
            processing_times.extend([avg_frame_time] * original_batch_length)

            # Update progress bar with current FPS
            current_fps = original_batch_length / frame_time if frame_time > 0 else 0
            pbar.set_postfix({'FPS': f'{current_fps:.1f}'})

            # Clear batch
            batch_frames = []

finally:
    # Close progress bar
    pbar.close()

    if processing_times:
        # Calculate and display final statistics
        total_time = time.time() - start_time
        average_fps = total_frames / total_time
        avg_time_per_frame = sum(processing_times) / len(processing_times)

        print("\nProcessing Complete!")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average FPS: {average_fps:.2f}")
        print(f"Average time per frame: {avg_time_per_frame*1000:.1f}ms")
        print(f"Total frames processed: {len(processing_times)}")
        print(f"Output saved to: {output_path}")

    # Clean up
    cap.release()
    writer.release()
    cv2.destroyAllWindows()