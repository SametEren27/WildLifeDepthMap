import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class InferenceEngine:
    def __init__(self, detector_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(">>> YOLO Yükleniyor...")
        self.detector = torch.hub.load('ultralytics/yolov5', 'custom', path=detector_path)
        
        print(">>> Depth Anything V2 Yükleniyor...")
        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        print(">>> MODELLER HAZIR.")

    def run_inference(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.image_processor(images=img_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()
  
        if callable(self.detector):
            results = self.detector(frame)
        else:
            results = self.detector.predict(frame) 

        df = results.pandas().xyxy[0]
          
        detections = []
        for index, row in df.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            conf = row['confidence']
            cls = row['class']
            
            detections.append([xmin, ymin, xmax, ymax, conf, cls])

        return depth_map, detections