import os
import time
import glob
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from models.lightning_tracker import SiamTrackerLightning

def get_iou(box1, box2):
    # box format: [x, y, w, h]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    
    if union == 0:
        return 0.0
    return inter / union

def get_center_error(box1, box2):
    # center coordinates
    c1 = np.array([box1[0] + box1[2]/2, box1[1] + box1[3]/2])
    c2 = np.array([box2[0] + box2[2]/2, box2[1] + box2[3]/2])
    return np.linalg.norm(c1 - c2)

def crop_and_resize_eval(image, cx, cy, s, output_sz):
    """
    Utility crop function for evaluation.
    Crops a patch of size s centered at (cx, cy) from image and resizes it to output_sz.
    """
    img_h, img_w, _ = image.shape
    
    x1 = int(cx - s / 2)
    y1 = int(cy - s / 2)
    x2 = int(cx + s / 2)
    y2 = int(cy + s / 2)
    
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)
    
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(img_w, x2)
    crop_y2 = min(img_h, y2)
    
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        avg_color = np.mean(image, axis=(0, 1)).astype(np.uint8).tolist()
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=avg_color
        )
        
    resized = cv2.resize(cropped, (output_sz, output_sz))
    return resized, cx - s / 2, cy - s / 2

def evaluate_model(model_path, seq_dir):
    """
    Run tracking on a sequence and calculate metrics.
    """
    # Load model
    print(f"Loading checkpoint {model_path}...")
    try:
        model = SiamTrackerLightning.load_from_checkpoint(model_path, student_pretrained=False)
    except Exception as e:
        print(f"Error loading checkpoint directly, attempting to load state dict: {e}")
        # Create student skeleton and load
        model = SiamTrackerLightning(student_pretrained=False)
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Read sequence image paths
    img_dir = os.path.join(seq_dir, "img")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + 
                       glob.glob(os.path.join(img_dir, "*.png")))
    
    # Read ground truth bboxes
    gt_file = os.path.join(seq_dir, "groundtruth_rect.txt")
    bboxes_gt = []
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for sep in [',', '\t', ' ']:
                parts = [p.strip() for p in line.split(sep) if p.strip()]
                if len(parts) >= 4:
                    break
            if len(parts) >= 4:
                bboxes_gt.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
                
    num_frames = min(len(img_paths), len(bboxes_gt), 50)
    if num_frames == 0:
        raise ValueError(f"No frames or groundtruth found in {seq_dir}")
        
    # Standard preprocessing
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Tracking results
    pred_boxes = []
    ious = []
    center_errors = []
    inference_times = []
    
    # Initial target box
    init_box = bboxes_gt[0]
    pred_boxes.append(init_box)
    ious.append(1.0)
    center_errors.append(0.0)
    
    # Load and process first frame to initialize template
    img = cv2.imread(img_paths[0])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract template patch using template crop logic
    x, y, w, h = init_box
    cx, cy = x + w / 2, y + h / 2
    p = (w + h) / 2
    s_z = np.sqrt((w + 2 * p) * (h + 2 * p))
    
    template_patch, _, _ = crop_and_resize_eval(img_rgb, cx, cy, s_z, output_sz=127)
    template_tensor = (template_patch.astype(np.float32) / 255.0 - mean) / std
    template_tensor = torch.from_numpy(template_tensor).permute(2, 0, 1).unsqueeze(0)
    
    if torch.cuda.is_available():
        template_tensor = template_tensor.cuda()
        
    # Initialize template features in student tracker
    model.student.init_template(template_tensor)
    
    # Run tracking loop
    curr_cx, curr_cy = cx, cy
    for f_idx in range(1, num_frames):
        img = cv2.imread(img_paths[f_idx])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop search patch centered around last target position
        s_search = s_z * (255.0 / 127.0)
        search_patch, crop_left, crop_top = crop_and_resize_eval(img_rgb, curr_cx, curr_cy, s_search, output_sz=255)
        search_tensor = (search_patch.astype(np.float32) / 255.0 - mean) / std
        search_tensor = torch.from_numpy(search_tensor).permute(2, 0, 1).unsqueeze(0)
        
        if torch.cuda.is_available():
            search_tensor = search_tensor.cuda()
            
        # Track (eval time)
        start_time = time.time()
        with torch.no_grad():
            cls_logits, reg_offsets = model.student.track(search_tensor)
        inference_times.append(time.time() - start_time)
        
        # Decode prediction on 17x17 map
        score_map = torch.sigmoid(cls_logits)[0, 0].cpu().numpy() # [17, 17]
        reg_map = reg_offsets[0].cpu().numpy()                   # [4, 17, 17]
        
        # Find position of max confidence
        j_max, i_max = np.unravel_index(np.argmax(score_map), score_map.shape)
        
        # Target offsets at max position
        l, t, r, b = reg_map[:, j_max, i_max]
        
        # Coordinates in 255x255 search space
        x_c_scaled = 64 + i_max * 8
        y_c_scaled = 64 + j_max * 8
        
        x_pred_scaled = x_c_scaled - l
        y_pred_scaled = y_c_scaled - t
        w_pred_scaled = l + r
        h_pred_scaled = t + b
        
        # Map back to original image space
        scale = 255.0 / s_search
        w_pred = w_pred_scaled / scale
        h_pred = h_pred_scaled / scale
        x_pred = crop_left + x_pred_scaled / scale
        y_pred = crop_top + y_pred_scaled / scale
        
        pred_box = [x_pred, y_pred, w_pred, h_pred]
        pred_boxes.append(pred_box)
        
        # Update center coordinates for the next search crop
        curr_cx = x_pred + w_pred / 2
        curr_cy = y_pred + h_pred / 2
        
        # Calculate frame metrics
        gt_box = bboxes_gt[f_idx]
        ious.append(get_iou(pred_box, gt_box))
        center_errors.append(get_center_error(pred_box, gt_box))
        
    mean_iou = np.mean(ious)
    success_rate_auc = np.mean(np.array(ious) >= 0.5) # Success at IoU >= 0.5
    precision_20px = np.mean(np.array(center_errors) <= 20.0)
    avg_fps = 1.0 / np.mean(inference_times) if len(inference_times) > 0 else 0.0
    
    return {
        "mean_iou": mean_iou,
        "success_auc": success_rate_auc,
        "precision_20px": precision_20px,
        "fps": avg_fps,
        "num_frames": num_frames
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Unpruned vs. Pruned Siamese Tracker")
    parser.add_argument("--unpruned_ckpt", type=str, default="weights/best_siam_tracker.ckpt", help="Path to unpruned model checkpoint")
    parser.add_argument("--pruned_ckpt", type=str, default="weights/best_siam_tracker_pruned.ckpt", help="Path to pruned model checkpoint")
    parser.add_argument("--seq_dir", type=str, default="data/Car1", help="Path to evaluation sequence folder")
    args = parser.parse_args()
    
    if not os.path.exists(args.seq_dir):
        print(f"Evaluation sequence directory {args.seq_dir} does not exist. Please download it first.")
        return
        
    print("\n==========================================")
    print("      EVALUATING UNPRUNED MODEL")
    print("==========================================")
    if os.path.exists(args.unpruned_ckpt):
        unpruned_results = evaluate_model(args.unpruned_ckpt, args.seq_dir)
        for k, v in unpruned_results.items():
            print(f"{k}: {v:.4f}" if k != "num_frames" else f"{k}: {v}")
    else:
        print(f"Unpruned checkpoint {args.unpruned_ckpt} not found.")
        unpruned_results = None
        
    print("\n==========================================")
    print("       EVALUATING PRUNED MODEL")
    print("==========================================")
    if os.path.exists(args.pruned_ckpt):
        pruned_results = evaluate_model(args.pruned_ckpt, args.seq_dir)
        for k, v in pruned_results.items():
            print(f"{k}: {v:.4f}" if k != "num_frames" else f"{k}: {v}")
    else:
        print(f"Pruned checkpoint {args.pruned_ckpt} not found.")
        pruned_results = None
        
    # Print comparison table
    if unpruned_results or pruned_results:
        print("\n========================================================")
        print("                 COMPARISON SUMMARY")
        print("========================================================")
        print(f"{'Metric':<20} | {'Unpruned Model':<16} | {'Pruned Model':<16}")
        print("-" * 60)
        
        metrics = ["mean_iou", "success_auc", "precision_20px", "fps"]
        for m in metrics:
            val_u = f"{unpruned_results[m]:.4f}" if unpruned_results and m in unpruned_results else "N/A"
            val_p = f"{pruned_results[m]:.4f}" if pruned_results and m in pruned_results else "N/A"
            print(f"{m:<20} | {val_u:<16} | {val_p:<16}")
        print("========================================================")

if __name__ == "__main__":
    main()
