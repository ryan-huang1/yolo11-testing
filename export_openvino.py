from ultralytics import YOLO
import time
import os

def export_model(model_path="yolo11n.pt", 
                imgsz=416,
                batch_size=1,
                simplify=True):
    
    print(f"Starting OpenVINO export of {model_path}...")
    start_time = time.time()
    
    try:
        # Load the model
        model = YOLO(model_path)
        
        # Export the model to OpenVINO
        model.export(format="openvino",
                    imgsz=imgsz,
                    batch=batch_size,
                    simplify=simplify)
        
        export_time = time.time() - start_time
        print(f"\nExport completed successfully in {export_time:.2f} seconds!")
        export_path = model_path.replace('.pt', '_openvino_model')
        print(f"OpenVINO model saved in directory: {export_path}")
        
        # Verify the exported model
        verify_model(export_path)
        
    except Exception as e:
        print(f"Error during export: {str(e)}")

def verify_model(model_path):
    """Verify the OpenVINO model can be loaded and used."""
    try:
        # Load and test the model
        test_model = YOLO(model_path)
        print("\nModel verification successful!")
        
        # Print model information
        print("\nModel Details:")
        print(f"Model path: {model_path}")
        
        # Calculate total size of the model directory
        total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(model_path)
                        for filename in filenames)
        print(f"Model size: {get_file_size(total_size)}")
        
    except Exception as e:
        print(f"Error during model verification: {str(e)}")

def get_file_size(size_bytes):
    """Get file size in a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

if __name__ == "__main__":
    # Export settings optimized for CPU (OpenVINO)
    settings = {
        'model_path': "yolo11n.pt",    
        'imgsz': 416,                  
        'batch_size': 1,               
        'simplify': True
    }
    
    # Export model
    export_model(**settings)