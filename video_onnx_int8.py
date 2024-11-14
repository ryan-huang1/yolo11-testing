from ultralytics import YOLO
import cv2
import time
from tqdm import tqdm

# Load YOLO model
model = YOLO("yolo11n_int8.onnx")

# Open video file
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize video writer
output_path = "output.mp4"
writer = cv2.VideoWriter(output_path,
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps,
                         (width, height))

# Performance tracking variables
start_time = time.time()
processing_times = []

# Model input size set to 320
input_size = 320  # Changed to 320x320

# Batch size
batch_size = 8

def pad_batch(batch_frames, batch_size):
    pad_size = batch_size - len(batch_frames)
    if pad_size > 0:
        last_frame = batch_frames[-1]
        for _ in range(pad_size):
            batch_frames.append(last_frame)
    return batch_frames

try:
    pbar = tqdm(total=total_frames, desc='Processing video', unit='frames')
    frame_count = 0
    batch_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        batch_frames.append(frame)
        frame_count += 1

        if len(batch_frames) == batch_size or frame_count == total_frames:
            if len(batch_frames) < batch_size:
                original_batch_length = len(batch_frames)
                batch_frames = pad_batch(batch_frames, batch_size)
            else:
                original_batch_length = batch_size

            frame_start_time = time.time()

            # Run inference with 320x320 image size
            results = model(batch_frames, imgsz=input_size)

            for result in results[:original_batch_length]:
                annotated_frame = result.plot()
                writer.write(annotated_frame)
                pbar.update(1)

            frame_time = time.time() - frame_start_time
            avg_frame_time = frame_time / original_batch_length
            processing_times.extend([avg_frame_time] * original_batch_length)

            current_fps = original_batch_length / frame_time if frame_time > 0 else 0
            pbar.set_postfix({'FPS': f'{current_fps:.1f}'})

            batch_frames = []

finally:
    pbar.close()

    if processing_times:
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