import os
import sys
import types
import numpy as np
import onnx

# 1. Metaprogramming Monkey Patch for NumPy 2.x allow_pickle compatibility
orig_load = np.load
def patched_load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    print(f"[np.load debug] type: {type(file)}, value: {str(file)[:200]}")
    is_npy = False
    
    # If input is a file path (string, bytes, or Pathlib object), open it and check the magic header
    if isinstance(file, (str, bytes)) or hasattr(file, '__fspath__'):
        try:
            with open(file, 'rb') as f:
                magic = f.read(6)
            print(f"[np.load debug] path magic: {magic}")
            if magic == b'\x93NUMPY':
                is_npy = True
        except Exception as e:
            print(f"[np.load debug] path check failed: {e}")
            
    # If input is a stream, check the magic header
    elif hasattr(file, 'tell') and hasattr(file, 'seek') and hasattr(file, 'read'):
        try:
            pos = file.tell()
            magic = file.read(6)
            file.seek(pos) # Reset immediately
            print(f"[np.load debug] stream magic at pos {pos}: {magic}")
            if magic == b'\x93NUMPY':
                is_npy = True
        except Exception as e:
            print(f"[np.load debug] stream check failed: {e}")
            
    # Force allow_pickle=True only for .npy files to prevent breaking raw buffers
    final_allow_pickle = allow_pickle or is_npy
    print(f"[np.load debug] resolved allow_pickle: {final_allow_pickle}")
    
    return orig_load(file, mmap_mode=mmap_mode, allow_pickle=final_allow_pickle, fix_imports=fix_imports, encoding=encoding)

np.load = patched_load
print("[NumPy Patch] Successfully registered debug numpy.load patch.")

# 2. Metaprogramming Monkey Patch for ONNX 1.20+ and NumPy 2.x pickling compatibility
TENSOR_TYPE_TO_NP_TYPE = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.UINT16: np.uint16,
    onnx.TensorProto.INT16: np.int16,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.STRING: np.object_,
    onnx.TensorProto.BOOL: np.bool_,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.UINT32: np.uint32,
    onnx.TensorProto.UINT64: np.uint64,
}

# Bind mock mapping module
mapping_module = types.ModuleType("onnx.mapping")
mapping_module.TENSOR_TYPE_TO_NP_TYPE = TENSOR_TYPE_TO_NP_TYPE
sys.modules["onnx.mapping"] = mapping_module
onnx.mapping = mapping_module
print("[ONNX Patch] Successfully mocked onnx.mapping module with raw NumPy constructors.")

# 3. Perform ONNX to TFLite conversion
from onnx2tf import convert

def main():
    onnx_path = "weights/best_siam_tracker_pruned.onnx"
    output_dir = "weights"
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file {onnx_path} not found.")
        return
        
    print(f"\n[onnx2tf] Starting programmatic conversion of {onnx_path}...")
    try:
        convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=output_dir,
            non_verbose=False
        )
        print("[onnx2tf] TFLite conversion completed.")
    except Exception as e:
        print(f"[onnx2tf] Conversion failed with error: {e}")

if __name__ == "__main__":
    main()
