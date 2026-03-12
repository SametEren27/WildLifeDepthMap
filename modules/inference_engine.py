# type: ignore
import torch
import cv2
import numpy as np

class InferenceEngine:
    def __init__(self, detector_path, device_type="cpu"):

        self.device = torch.device(device_type)
        
        print("Modeller yükleniyor, bu biraz zaman alabilir...")
        
        self.detector = torch.hub.load('ultralytics/yolov5', 'custom', path=detector_path)
        self.detector.to(self.device)
        self.detector.eval()

        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.midas.to(self.device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        
        print("Modeller başarıyla yüklendi ve cihazına (Device) bağlandı.")

    def run_inference(self, frame):

        results = self.detector(frame)
        detections = results.xyxy[0].cpu().numpy() 
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            depth_map = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        return depth_map, detections