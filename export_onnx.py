from ultralytics import YOLO
import time

def export_model(model_path="yolo11n.pt", 
                imgsz=416,
                batch_size=8,
                opset=12,  # ONNX opset version
                simplify=True,  # Simplify model
                dynamic=False,  # Use dynamic axes
                half=False):    # FP16 quantization
    
    print(f"Starting ONNX export of {model_path}...")
    start_time = time.time()
    
    try:
        # Load the model
        model = YOLO(model_path)
        
        # Export the model
        model.export(format="onnx",
                    imgsz=imgsz,
                    batch=batch_size,
                    opset=opset,
                    simplify=simplify,
                    dynamic=dynamic,
                    half=half)
        
        export_time = time.time() - start_time
        print(f"\nExport completed successfully in {export_time:.2f} seconds!")
        print(f"ONNX model saved as '{model_path.replace('.pt', '.onnx')}'")
        
        # Verify the exported model
        verify_model(model_path.replace('.pt', '.onnx'))
        
    except Exception as e:
        print(f"Error during export: {str(e)}")

def verify_model(onnx_path):
    """Verify the ONNX model can be loaded and used."""
    try:
        # Load and test the ONNX model
        test_model = YOLO(onnx_path)
        print("\nModel verification successful!")
        
        # Print model information
        print("\nModel Details:")
        print(f"Model path: {onnx_path}")
        print(f"Model size: {get_file_size(onnx_path)}")
        
    except Exception as e:
        print(f"Error during model verification: {str(e)}")

def get_file_size(file_path):
    """Get file size in a human-readable format."""
    size_bytes = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

if __name__ == "__main__":
    import os
    
    # Export settings
    settings = {
        'model_path': "yolo11n.pt",    # Path to your model
        'imgsz': 416,                  # Input size (smaller for CPU)
        'batch_size': 8,               # Batch size for CPU
        'opset': 12,                   # ONNX opset version
        'simplify': True,              # Simplify model
        'dynamic': False,              # Dynamic batch size
        'half': False                  # FP16 quantization (set False for CPU)
    }
    
    # Export model
    export_model(**settings)