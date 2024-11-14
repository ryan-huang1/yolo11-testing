from ultralytics import YOLO
import cv2
import time
from tqdm import tqdm

# Load YOLO model in NCNN format
model = YOLO("/Users/ryanhuang/Downloads/content/yolo11n_ncnn_model", task='detect')  # Change to your NCNN model path

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

# Model input size
input_size = 320  # Changed to 320 as it's typically better for NCNN

# Batch size
batch_size = 1  # NCNN typically works better with batch_size=1 on embedded devices

try:
    # Create progress bar
    pbar = tqdm(total=total_frames, desc='Processing video', unit='frames')
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_start_time = time.time()

        # Run inference (NCNN works best with single frames)
        results = model(frame, imgsz=input_size)

        # Visualize and write results
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)
        
        # Calculate processing time
        frame_time = time.time() - frame_start_time
        processing_times.append(frame_time)

        # Update progress bar with current FPS
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        pbar.set_postfix({'FPS': f'{current_fps:.1f}'})
        pbar.update(1)

finally:
    # Close progress bar
    pbar.close()

    if processing_times:  # Only calculate stats if we have processed frames
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