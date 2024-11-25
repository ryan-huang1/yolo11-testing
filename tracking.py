from ultralytics import YOLO
import cv2
import time
from tqdm import tqdm
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
                         fps // 2,  # Adjust FPS for skipping frames
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
MAX_TRAIL_LENGTH = 30
IOU_THRESHOLD = 0.3

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def update_trails(detections, frame_number):
    """Update car trails with new detections."""
    current_boxes = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        current_boxes.append((x1, y1, x2, y2))

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

    for i in range(len(current_boxes)):
        if i not in matched_detections:
            new_trail_id = len(car_trails)
            car_trails[new_trail_id].append((frame_number, current_boxes[i]))

    for trail_id in car_trails:
        if len(car_trails[trail_id]) > MAX_TRAIL_LENGTH:
            car_trails[trail_id] = car_trails[trail_id][-MAX_TRAIL_LENGTH:]

def draw_trails(frame, frame_number):
    """Draw trails on the frame with smoothed positions."""
    for trail_id, positions in car_trails.items():
        if positions and positions[-1][0] >= frame_number - MAX_TRAIL_LENGTH * 2:
            centers = []
            for pos in positions:
                box = pos[1]
                center = (
                    (box[0] + box[2]) // 2,
                    (box[1] + box[3]) // 2
                )
                centers.append(center)

            # Apply exponential smoothing
            smoothed_centers = []
            alpha_smoothing = 0.2  # Smoothing factor
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

            # Draw smoothed trails
            for i in range(1, len(smoothed_centers)):
                prev_center = smoothed_centers[i - 1]
                curr_center = smoothed_centers[i]
                overlay = frame.copy()
                alpha = 0.7 * (1 - (frame_number - positions[i][0]) / (MAX_TRAIL_LENGTH * 2))
                if alpha > 0:
                    cv2.line(overlay, prev_center, curr_center, (0, 255, 0), 2)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

try:
    total_processed_frames = total_frames // 2
    pbar = tqdm(total=total_processed_frames, desc='Processing video', unit='frames')

    frame_count = 0
    batch_frames = []
    frame_numbers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        batch_frames.append(frame)
        frame_numbers.append(frame_count)
        frame_count += 2  # Increment by 2 as we're skipping every other frame

        cap.grab()  # Skip the next frame

        if len(batch_frames) == batch_size or frame_count >= total_frames:
            results = model(batch_frames, imgsz=input_size)

            for i, result in enumerate(results):
                car_detections = result.boxes[result.boxes.cls == 2]
                update_trails(car_detections, frame_numbers[i])
                current_frame = batch_frames[i].copy()
                draw_trails(current_frame, frame_numbers[i])
                writer.write(current_frame)
                pbar.update(1)

            batch_frames = []
            frame_numbers = []

finally:
    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print("\nProcessing Complete!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Output saved to: {output_path}")
