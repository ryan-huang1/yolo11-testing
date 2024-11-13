from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("yolo11n.pt")

# Run inference
results = model("https://ultralytics.com/images/bus.jpg")

# Get the first result (since we only processed one image)
result = results[0]

# Plot the result
plotted_img = result.plot()

# Convert from BGR to RGB
plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

# Save the image
cv2.imwrite("bus_with_boxes.jpg", plotted_img)