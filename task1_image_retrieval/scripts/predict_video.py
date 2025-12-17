import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import json
import os
import sys

# Add the project root directory to the python path so we can import 'models'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.detector.rtdetr.rtdetr import RTDETR

# COCO Class Names (80 classes)
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Model Configuration (Same as usage example)
def get_model(config_path=None):
    if config_path is None:
        # Default to configs/rtdetr_config.json relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'configs', 'rtdetr_config.json')
    
    print(f"Loading model configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Helper function to convert list to tuple for specific keys if needed, 
    # but based on previous code, lists were used for most config items except maybe logic that expects tuples.
    # Looking at the original dictionary, mostly lists were used (e.g. 'return_idx': [1, 2, 3]).
    # So direct loading should be fine.
    
    weights_path = config.get('weights_path', 'weights/rtdetrv2_r34vd_120e_coco_ema.pth')
    
    print("Initializing model...")
    model = RTDETR(
        num_classes=80,
        backbone_conf=config['backbone_conf'],
        encoder_conf=config['encoder_conf'],
        decoder_conf=config['decoder_conf'],
        weights=weights_path
    )
    model.eval()
    return model

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = out_bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [0, 0, 255]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    model = get_model()
    model.to(device)

    video_path = 'video_sample.mp4'
    output_path = 'output_video.mp4'
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        # Note: RT-DETR might expect different normalization or none, checking utils? 
        # Defaulting to no normalization for now, just 0-1
    ])

    frame_count = 0
    
    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Preprocess
        img = transform(pil_img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(img)
            
        # Postprocess
        logits = output['pred_logits'] # [1, 300, 80]
        boxes = output['pred_boxes']   # [1, 300, 4]
        
        prob = logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(1, -1), 100, dim=1)
        scores = topk_values[0]
        topk_boxes = topk_indexes // 80
        labels = topk_indexes % 80
        
        boxes = box_cxcywh_to_xyxy(boxes)
        boxes = boxes[0, topk_boxes[0]]
        
        # Rescale boxes to original image size
        scaled_boxes = rescale_bboxes(boxes.cpu(), (width, height))
        
        # Draw boxes
        scores = scores.cpu()
        labels = labels.cpu()
        keep = scores > 0.5
        
        for box, score, label_idx in zip(scaled_boxes[keep], scores[keep], labels[0][keep]):
            if label_idx != 2:
                continue
            label = f"{CLASSES[label_idx]}: {score:.2f}"
            plot_one_box(box.tolist(), frame, label=label, color=[0, 255, 0])
            
        cv2.imshow('RT-DETR Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        out.write(frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved to {output_path}")

if __name__ == '__main__':
    main()
