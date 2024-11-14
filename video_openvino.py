from ultralytics import YOLO
import cv2
import time
from tqdm import tqdm
import numpy as np

# Load YOLO model
model = YOLO("yolo11n_openvino_model/")

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
input_size = 416

try:
    # Create progress bar
    pbar = tqdm(total=total_frames, desc='Processing video', unit='frames')
    
    while True:
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame for inference and ensure correct shape
        resized_frame = cv2.resize(frame, (input_size, input_size))
        
        # Convert to float32 and normalize to [0-1]
        input_frame = resized_frame.astype(np.float32) / 255.0
        
        # Ensure correct shape (1, 3, 416, 416)
        input_frame = input_frame.transpose(2, 0, 1)  # HWC to CHW format
        input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension
        
        # Verify shape before inference
        print(f"Input shape: {input_frame.shape}")  # Should be (1, 3, 416, 416)
        
        # Run inference with explicit size
        results = model(input_frame, imgsz=input_size)
        
        # Visualize results
        annotated_frame = results[0].plot()
        
        # Write frame to output video
        writer.write(annotated_frame)
        
        # Calculate frame processing time
        frame_time = time.time() - frame_start_time
        processing_times.append(frame_time)
        
        # Update progress bar with current FPS
        current_fps = 1 / frame_time if frame_time > 0 else 0
        pbar.set_postfix({'FPS': f'{current_fps:.1f}'})
        pbar.update(1)

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