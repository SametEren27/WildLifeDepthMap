# type: ignore
import os
import cv2
import torch
from modules.inference_engine import InferenceEngine
from modules.species_classifier import SpeciesClassifier

DETECTOR_PATH = r'C:\WildLifeDepthMap\models\md_v5b.0.0.pt' 
IMAGE_FOLDER = 'samples/' 
DEVICE = 'cpu' 

def main():
    engine = InferenceEngine(DETECTOR_PATH, DEVICE)

    classifier = SpeciesClassifier()

    images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.JPG', '.jpeg', '.png'))]
    print(f"{len(images)} adet fotoğraf bulundu. İşlem başlıyor...")

    for img_name in images:
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        frame = cv2.imread(img_path)
        if frame is None: continue

        depth_map, detections = engine.run_inference(frame)
        
        depth_display = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        print(f"\nResim: {img_name} | Tespit: {len(detections)}")

        for i, det in enumerate(detections):
            xmin, ymin, xmax, ymax, conf, cls = det
            
            x_target = int((xmin + xmax) / 2)
            y_target = int(ymax)

            y_target = min(y_target, depth_map.shape[0] - 1)
            x_target = min(x_target, depth_map.shape[1] - 1)
            
            depth_value = depth_map[y_target, x_target]
            
            animal_crop = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

            if animal_crop.size > 0:
                species_name, species_conf = classifier.predict(animal_crop)
            else:
                species_name, species_conf = "Unknown", 0

            # print(f"  - Geyik {i+1}: %{conf*100:.1f}, Derinlik: {depth_value:.4f}")

            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)           
            # 1. Define the dynamic scale (Keep this part)
            base_width = 2000 
            dynamic_scale = frame.shape[1] / base_width
            dynamic_thickness = max(2, int(3 * dynamic_scale))

            # 2. Create the label string
            label_str = f"{species_name} %{species_conf*100:.1f} | Depth: {depth_value:.2f}"

            # 3. Calculate text box size
            (w, h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, dynamic_scale, dynamic_thickness)

            # 4. Draw the background box (Filled Black)
            # We use h+20 to give it some "padding"
            cv2.rectangle(frame, (int(xmin), int(ymin) - h - 40), (int(xmin) + w, int(ymin)), (0, 0, 0), -1)

            # 5. Draw the text (Green) on top of the black box
            cv2.putText(frame, label_str, (int(xmin), int(ymin) - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, dynamic_scale, (0, 255, 0), dynamic_thickness)
        combined_view = cv2.hconcat([cv2.resize(frame, (640, 480)), cv2.resize(depth_display, (640, 480))])
        cv2.imshow('WildLife Detection & Depth', combined_view)
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()