# type: ignore
import os
import cv2
import torch
import numpy as np
import math
from modules.inference_engine import InferenceEngine
from modules.species_classifier import SpeciesClassifier

DETECTOR_PATH = r'C:\WildLifeDepthMap\models\md_v5b.0.0.pt' 
IMAGE_FOLDER = 'specifictest/' 
DEVICE = 'cpu' 

K_SLOPE = 60 
OFF_SET = -0.5  
H_METERS = 0.5   
V_FOV = 42.0    

def get_depth_at_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_map = param
        try:
            val = depth_map[y, x]
            print(f"\n[CLICK] X:{x} Y:{y} | RAW DEPTH: {val:.4f}")
        except Exception as e:
            print(f"Click error: {e}")


def calibrate_camera(distances, depths):
    """
    Call this if you want to find your custom K_SLOPE and OFF_SET.
    distances: e.g. [3, 5, 10] | depths: e.g. [17.0, 24.3, 11.0]
    """
    if len(distances) < 2: return 100.0, 0.0
    inv_depths = [1.0 / d for d in depths]
    m, c = np.polyfit(inv_depths, distances, 1)
    print(f"New Calibration: Slope(K)={m:.2f}, Offset(C)={c:.2f}")
    return m, c


def main():
    engine = InferenceEngine(DETECTOR_PATH, DEVICE)
    classifier = SpeciesClassifier()

    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"{len(images)} adet fotoğraf bulundu. İşlem başlıyor...")

    for img_name in images:
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        frame = cv2.imread(img_path)
        if frame is None: continue

        info_bar_h = int(frame.shape[0] / 21)
        frame = frame[0:frame.shape[0]-info_bar_h, :]

        depth_map, detections = engine.run_inference(frame)

        depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        print(f"\nResim: {img_name} | Tespit: {len(detections)}")

        for i, det in enumerate(detections):
            xmin, ymin, xmax, ymax, conf, cls = det
            
            animal_crop = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
            species_name, species_conf = ("Unknown", 0)
            if animal_crop.size > 0:
                species_name, species_conf = classifier.predict(animal_crop)

            roi_h = int((ymax - ymin) / 6)
            roi = depth_map[max(0, int(ymax - roi_h)):int(ymax), int(xmin):int(xmax)]
            
            if roi.size > 0:
                robust_depth = np.percentile(roi, 95)
            else:
                robust_depth = 1.0

            dist_ai = (K_SLOPE / robust_depth) + OFF_SET
            
            img_h = frame.shape[0]
            rel_y = (ymax - (img_h / 2)) / (img_h / 2)
            angle_rad = math.radians(rel_y * (V_FOV / 2))
            dist_geo = H_METERS / math.tan(angle_rad) if angle_rad > 0 else dist_ai
            
            final_dist = (dist_ai * 0.8) + (dist_geo * 0.2)

            base_w = 2000
            dyn_scale = frame.shape[1] / base_w
            dyn_thick = max(2, int(3 * dyn_scale))

            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.rectangle(frame, (int(xmin), int(ymax - roi_h)), (int(xmax), int(ymax)), (0, 0, 255), 2)

            label_str = f"{species_name} {species_conf*100:.0f}% | {final_dist:.1f}m (Raw:{robust_depth:.1f})"
            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, dyn_scale, dyn_thick)
            
            cv2.rectangle(frame, (int(xmin), int(ymin) - th - 20), (int(xmin) + tw, int(ymin)), (0, 0, 0), -1)
            cv2.putText(frame, label_str, (int(xmin), int(ymin) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, dyn_scale, (0, 255, 0), dyn_thick)

        win_name = 'WildLife Detection & Depth'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) 
        cv2.setMouseCallback(win_name, get_depth_at_click, depth_map)
        
        cv2.imshow(win_name, frame)
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()