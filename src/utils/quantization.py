import os
import argparse
import numpy as np
import torch
import cv2
from models.lightning_tracker import SiamTrackerLightning

def get_representative_dataset(seq_dir, num_samples=100):
    """
    Generator of representative samples from the dataset for TFLite quantization calibration.
    """
    import glob
    img_dir = os.path.join(seq_dir, "img")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + 
                       glob.glob(os.path.join(img_dir, "*.png")))
    
    if len(img_paths) == 0:
        # Fallback to random data if no images found
        def representative_data_gen():
            for _ in range(num_samples):
                yield [
                    np.random.randn(1, 3, 127, 127).astype(np.float32),
                    np.random.randn(1, 3, 255, 255).astype(np.float32)
                ]
        return representative_data_gen
        
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    num_samples = min(num_samples, len(img_paths))
    
    def representative_data_gen():
        for i in range(num_samples):
            img = cv2.imread(img_paths[i])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Simple crops for calibration
            h_img, w_img, _ = img_rgb.shape
            sz_z = min(h_img, w_img, 127)
            sz_x = min(h_img, w_img, 255)
            
            crop_z = cv2.resize(img_rgb[:sz_z, :sz_z], (127, 127))
            crop_x = cv2.resize(img_rgb[:sz_x, :sz_x], (255, 255))
            
            tensor_z = (crop_z.astype(np.float32) / 255.0 - mean) / std
            tensor_x = (crop_x.astype(np.float32) / 255.0 - mean) / std
            
            tensor_z = np.expand_dims(np.transpose(tensor_z, (2, 0, 1)), axis=0)
            tensor_x = np.expand_dims(np.transpose(tensor_x, (2, 0, 1)), axis=0)
            
            yield [tensor_z, tensor_x]
            
    return representative_data_gen

def export_onnx(model, onnx_path):
    print(f"Exporting model to ONNX format at {onnx_path}...")
    dummy_z = torch.randn(1, 3, 127, 127)
    dummy_x = torch.randn(1, 3, 255, 255)
    
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_z = dummy_z.cuda()
        dummy_x = dummy_x.cuda()
        
    torch.onnx.export(
        model.student,
        (dummy_z, dummy_x),
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['template', 'search'],
        output_names=['cls_logits', 'reg_offsets']
    )
    print("ONNX export completed successfully.")

def export_tflite_ai_edge(model, tflite_path):
    """
    Attempt to convert the PyTorch model directly to TFLite using ai_edge_torch.
    """
    try:
        import ai_edge_torch
        print(f"Converting model to TFLite using ai_edge_torch...")
        
        # ai_edge_torch requires CPU-bound model for tracing
        model = model.cpu()
        model.student.eval()
        
        dummy_z = torch.randn(1, 3, 127, 127)
        dummy_x = torch.randn(1, 3, 255, 255)
        
        edge_model = ai_edge_torch.convert(model.student, (dummy_z, dummy_x))
        edge_model.export(tflite_path)
        print(f"Successfully saved TFLite model to {tflite_path}")
        return True
    except ImportError:
        print("ai_edge_torch is not installed. Skipping direct PyTorch-to-TFLite conversion.")
        return False
    except Exception as e:
        print(f"Failed to convert using ai_edge_torch: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Export Trained Siamese Tracker to ONNX and TFLite")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to PyTorch model checkpoint (.ckpt)")
    parser.add_argument("--output_dir", type=str, default="weights", help="Directory to save exported models")
    parser.add_argument("--seq_dir", type=str, default="data/Car1", help="Path to sequence directory for calibration")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.basename(args.ckpt_path).replace(".ckpt", "")
    onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
    tflite_path = os.path.join(args.output_dir, f"{base_name}.tflite")
    
    # Load model
    print(f"Loading checkpoint {args.ckpt_path}...")
    try:
        model = SiamTrackerLightning.load_from_checkpoint(args.ckpt_path)
    except Exception as e:
        print(f"Error loading checkpoint directly, loading state dict: {e}")
        model = SiamTrackerLightning()
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    
    # 1. Export to ONNX (Standard deployment fallback)
    export_onnx(model, onnx_path)
    
    # 2. Convert to TFLite (Direct via Google ai_edge_torch)
    converted = export_tflite_ai_edge(model, tflite_path)
    
    if not converted:
        print("\nNote: Direct TFLite export was not completed. You can convert the exported ONNX model to TFLite using 'onnx2tf':")
        print(f"  pip install onnx2tf tensorflow")
        print(f"  onnx2tf -i {onnx_path} -o {args.output_dir}")
        print("This tool supports 8-bit integer quantization using representative calibration data.")

if __name__ == "__main__":
    main()
