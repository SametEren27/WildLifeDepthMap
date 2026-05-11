import os
import pandas as pd
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import re

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

class BatchProcessor:
    def __init__(self, engine, cpp_lib=None):
        self.engine = engine
        self.cpp_lib = cpp_lib

    # DİKKAT: process_folder artık sınıfın (class) içinde!
    def process_folder(self, input_dir, entries_data, filter_settings, progress_callback):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(input_dir, f"Analiz_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        results = []

        # Kalibrasyon verilerini hazırla
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

                        animal_roi = d_map[iymin:iymax, ixmin:ixmax]
                        
                        if animal_roi.size > 0:
                            d_final = np.median(animal_roi)
                            animal_std = np.std(animal_roi)
                            batch_conf = max(0, min(100, 100 - (animal_std * 500)))
                        else:
                            d_final, batch_conf = 0.5, 0

                        estimated_dist = np.interp(d_final, x_calib, y_calib) if len(x_calib) >= 2 else 0.0
                        exif = get_exif_data(img_path)

                        # Temel satırı oluştur
                        row = {
                            "Dosya Adı": clean_string(filename),
                            "Tahmini Mesafe (m)": round(estimated_dist, 3),
                            "Güven Oranı (%)": round(batch_conf, 2),
                            "Ham AI Değeri": round(float(d_final), 5),
                            "Tespit Skoru": round(float(conf), 2),
                            "İşlem Zamanı": datetime.now().strftime("%H:%M:%S")
                        }

                        # EXIF verilerini row içine ekle
                        for tag, value in exif.items():
                            clean_tag = clean_string(str(tag))
                            clean_val = clean_string(str(value))
                            if clean_tag not in row:
                                row[clean_tag] = clean_val
                        
                        # --- DOĞRU YER: EXIF döngüsü bitti, şimdi append yapıyoruz ---
                        results.append(row)

                else:
                    # Hayvan bulunamadıysa tek satır ekle
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