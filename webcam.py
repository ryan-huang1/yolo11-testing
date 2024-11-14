from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Variables for FPS calculation
prev_time = 0
fps = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Run inference
        results = model(frame)
        
        # Visualize results
        annotated_frame = results[0].plot()
        
        # Add FPS text to frame
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        
        # Display
        cv2.imshow('YOLO11 Webcam (Press Q to Quit)', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()