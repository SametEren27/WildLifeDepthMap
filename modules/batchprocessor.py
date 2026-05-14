import os
import pandas as pd
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import re
from scipy.interpolate import interp1d

def clean_string(value):
    """Excel'in kabul etmediği illegal karakterleri temizler."""
    if not isinstance(value, str):
        return value
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)

def get_exif_data(path):
    """Resmin içindeki tüm metadataları (GPS dahil) ayıklar."""
    metadata = {}
    try:
        img = Image.open(path)
        exif_info = img.getexif()
        if exif_info:
            for tag_id, value in exif_info.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    gps_data = {}
                    for g_tag_id in value:
                        g_tag = GPSTAGS.get(g_tag_id, g_tag_id)
                        gps_data[g_tag] = value[g_tag_id]
                    metadata["GPS"] = str(gps_data)
                else:
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', 'ignore').strip()
                        except:
                            value = str(value)
                    metadata[tag] = value
    except Exception as e:
        print(f"Metadata okuma hatası: {e}")
    return metadata

def calculate_distance(raw_val, x_ref, y_ref, mode="log"):
    """Tavanı yıkan logaritmik mesafe hesaplama."""
    if len(x_ref) < 2: return 0.0
    pairs = sorted(zip(x_ref, y_ref))
    x_s = np.array([p[0] for p in pairs], dtype=float)
    y_s = np.array([p[1] for p in pairs], dtype=float)
    
    x_s = np.maximum(x_s, 1e-6)
    y_s = np.maximum(y_s, 1e-6)
    raw_val = max(raw_val, 1e-6)

    try:
        if mode == "log":
            model = interp1d(np.log(x_s), np.log(y_s), kind='linear', fill_value="extrapolate", bounds_error=False) # type: ignore
            dist = np.exp(float(model(np.log(raw_val))))
        else:
            model = interp1d(x_s, y_s, kind='linear', fill_value="extrapolate", bounds_error=False) # type: ignore
            dist = float(model(raw_val))
        return max(0.1, dist)
    except:
        return 0.0

class BatchProcessor:
    def __init__(self, engine, cpp_lib=None):
        self.engine = engine
        self.cpp_lib = cpp_lib

    def process_folder(self, input_dir, entries_data, filter_settings, progress_callback):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(input_dir, f"Analiz_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        results = []

        pairs = sorted([(float(v), float(m)) for m, v in entries_data.items() if v])
        x_calib = [p[0] for p in pairs] if pairs else []
        y_calib = [p[1] for p in pairs] if pairs else []

        print(f">>> {len(images)} resim için işlem başlıyor...")

        for i, filename in enumerate(images):
            img_path = os.path.join(input_dir, filename)
            try:
                raw_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if raw_img is None: continue

                d_map, detections = self.engine.run_inference(raw_img)
                h_orig, w_orig = d_map.shape[:2]

                if detections:
                    for det in detections:
                        xmin, ymin, xmax, ymax, conf, cls = det
                        ixmin, iymin, ixmax, iymax = int(xmin), int(ymin), int(xmax), int(ymax)
                        ixmin, iymin = max(0, ixmin), max(0, iymin)
                        ixmax, iymax = min(w_orig, ixmax), min(h_orig, iymax)
                        
                        box_h = iymax - iymin
                        
                        is_vertical = box_h > (ixmax - ixmin) * 1.5
                        roi_ratio = 0.20 if is_vertical else 0.30
                        
                        roi = d_map[int(iymax - box_h * roi_ratio):iymax, ixmin:ixmax]
                        
                        if roi.size > 0:
                            d_final = np.median(roi)
                            animal_std = np.std(roi)
                            batch_conf = max(0, min(100, 100 - (animal_std * 500)))
                        else:
                            d_final, batch_conf = 0.5, 0

                        estimated_dist = calculate_distance(d_final, x_calib, y_calib, mode="log")
                        
                        exif = get_exif_data(img_path)

                        row = {
                            "Dosya Adı": clean_string(filename),
                            "Tahmini Mesafe (m)": round(estimated_dist, 3),
                            "Güven Oranı (%)": round(batch_conf, 2),
                            "Ham AI Değeri": round(float(d_final), 5),
                            "Tespit Skoru": round(float(conf), 2),
                            "Tespit Tipi": "DIKEY" if is_vertical else "YATAY",
                            "İşlem Zamanı": datetime.now().strftime("%H:%M:%S")
                        }

                        for tag, value in exif.items():
                            clean_tag = clean_string(str(tag))
                            clean_val = clean_string(str(value))
                            if clean_tag not in row:
                                row[clean_tag] = clean_val
                        
                        results.append(row)

                else:
                    results.append({
                        "Dosya Adı": filename, 
                        "Tahmini Mesafe (m)": "Tespit Yok", 
                        "Güven Oranı (%)": 0,
                        "İşlem Zamanı": datetime.now().strftime("%H:%M:%S")
                    })

            except Exception as e:
                print(f"Hata ({filename}): {e}")

            progress_callback(i + 1, len(images))

        if results:
            df = pd.DataFrame(results)
            excel_path = os.path.join(output_dir, "analiz_raporu.xlsx")
            df.to_excel(excel_path, index=False)
            print(f">>> Analiz tamamlandı. Dosya: {excel_path}")
            
        return output_dir