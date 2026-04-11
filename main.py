# type: ignore
import os
import cv2
import torch
import numpy as np
import math
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sys

from modules.inference_engine import InferenceEngine
from modules.species_classifier import SpeciesClassifier

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

DEVICE = 'cpu'
DETECTOR_PATH = get_resource_path(os.path.join('models', 'md_v5b.0.0.pt'))

class WildlifeMetricPrototype(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Wildlife Metric Lab v2.9 - Flexible Window")
        self.geometry("950x980")
        
        self.engine = None
        self.last_depth_map = None
        self.ref_path = None
        self.animal_path = None
        self.calib_win = None

        try:
            if os.path.exists(DETECTOR_PATH):
                self.engine = InferenceEngine(DETECTOR_PATH, DEVICE)
                print(">>> MOTOR HAZIR.")
        except Exception as e:
            print(f">>> MOTOR HATASI: {e}")

        self.setup_ui()

    def setup_ui(self):
        # 1. DOSYA SEÇİMİ
        f_frame = ctk.CTkFrame(self)
        f_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(f_frame, text="1. REFERANS SEÇ", command=self.select_ref).pack(side="left", padx=10, pady=10, expand=True)
        ctk.CTkButton(f_frame, text="2. ANALİZ SEÇ", command=self.select_animal).pack(side="left", padx=10, pady=10, expand=True)

        # 2. DEĞER PANELI
        self.info_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.info_frame.pack(pady=5, padx=20, fill="x")
        self.click_label = ctk.CTkLabel(self.info_frame, text="GÜNCEL AI DEĞERİ: ---", font=("Arial", 18, "bold"), text_color="#FFCC00")
        self.click_label.pack(pady=10)

        # 3. MESAFE GİRİŞLERİ
        self.ref_frame = ctk.CTkFrame(self)
        self.ref_frame.pack(pady=10, padx=20, fill="both")
        self.meters = [1, 3, 5, 7, 9, 11, 13, 15]
        self.entries = {}
        for i, m in enumerate(self.meters):
            r, c = divmod(i, 2)
            row = ctk.CTkFrame(self.ref_frame); row.grid(row=r, column=c, padx=10, pady=5, sticky="ew")
            ctk.CTkLabel(row, text=f"{m}m Değeri:", width=80).pack(side="left")
            ent = ctk.CTkEntry(row, width=120); ent.pack(side="right", padx=5)
            self.entries[m] = ent

        # 4. YATAY TELAFİ
        h_frame = ctk.CTkFrame(self, fg_color="#2b2b2b")
        h_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(h_frame, text="YATAY TELAFİ (3m Çubukları)", font=("Arial", 12, "bold")).pack(pady=2)
        h_row = ctk.CTkFrame(h_frame, fg_color="transparent")
        h_row.pack(pady=5)
        ctk.CTkLabel(h_row, text="Sol X:").pack(side="left", padx=5)
        self.ent_x_l = ctk.CTkEntry(h_row, width=80); self.ent_x_l.pack(side="left", padx=5)
        ctk.CTkLabel(h_row, text="Sağ X:").pack(side="left", padx=5)
        self.ent_x_r = ctk.CTkEntry(h_row, width=80); self.ent_x_r.pack(side="left", padx=5)

        # 5. FORMÜL SEÇİMİ
        self.formula_option = ctk.CTkComboBox(self, values=["Linear Interpolation", "2. Degree Polynomial"])
        self.formula_option.pack(pady=10); self.formula_option.set("Linear Interpolation")

        # 6. BUTONLAR
        self.calib_btn = ctk.CTkButton(self, text="KALİBRASYON EKRANINI AÇ", command=self.open_calibration, state="disabled")
        self.calib_btn.pack(pady=5)
        self.run_btn = ctk.CTkButton(self, text="ANALİZİ BAŞLAT", command=self.start_analysis, fg_color="green", height=45, state="disabled")
        self.run_btn.pack(pady=20)

    def select_ref(self):
        self.ref_path = filedialog.askopenfilename()
        if self.ref_path: self.calib_btn.configure(state="normal", fg_color="blue")

    def select_animal(self):
        self.animal_path = filedialog.askopenfilename()
        if self.animal_path: self.run_btn.configure(state="normal")

    def open_calibration(self):
        if self.calib_win is not None and self.calib_win.winfo_exists():
            self.calib_win.focus(); return
        
        frame = cv2.imread(self.ref_path)
        depth_map, _ = self.engine.run_inference(frame)
        self.last_depth_map = depth_map
        self.orig_h, self.orig_w = frame.shape[:2]

        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_jet = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        # Orijinal resim ve derinlik haritasını yan yana birleştir
        combined = np.hstack((frame, depth_jet)) # Bu sefer resize etmeden ham birleştiriyoruz
        
        img_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        self.calib_win = ctk.CTkToplevel(self)
        self.calib_win.title("Kalibrasyon Ekranı")
        self.calib_win.geometry("1300x600")

        # ÖNEMLİ: CTkImage kullanarak resmi pencereye göre ölçeklenebilir yapıyoruz
        self.ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(1280, 500))
        
        self.img_label = ctk.CTkLabel(self.calib_win, image=self.ctk_img, text="")
        self.img_label.pack(fill="both", expand=True, pady=10, padx=10)
        
        # Tıklama olayını bağla
        self.img_label.bind("<Button-1>", self.on_tkinter_click)

    def on_tkinter_click(self, event):
        # Pencerenin O ANKİ gerçek boyutlarını al (Dinamik Ölçekleme)
        curr_width = self.img_label.winfo_width()
        curr_height = self.img_label.winfo_height()
        
        # Pencere iki resimden oluştuğu için tek resim genişliği yarısıdır
        one_img_width = curr_width / 2
        
        # Tıklanan x koordinatını sağ mı sol mu kontrol et
        x = event.x
        clicked_x = x if x < one_img_width else x - one_img_width
        
        # --- DİNAMİK EŞLEME FORMÜLÜ ---
        # (Tıklanan Piksel / O anki Genişlik) * Orijinal Resim Genişliği
        real_x = int((clicked_x / one_img_width) * self.orig_w)
        real_y = int((event.y / curr_height) * self.orig_h)
        
        # Sınır Güvenliği
        real_x = max(0, min(real_x, self.orig_w - 1))
        real_y = max(0, min(real_y, self.orig_h - 1))
        
        val = self.last_depth_map[real_y, real_x]
        self.click_label.configure(text=f"AI DEĞERİ: {val:.6f} | X: {real_x}")

    def start_analysis(self):
        try:
            x_ref, y_ref = [], []
            for m, ent in self.entries.items():
                if ent.get():
                    x_ref.append(float(ent.get()))
                    y_ref.append(float(m))
            
            if len(x_ref) < 2:
                messagebox.showwarning("Hata", "Referansları girin!")
                return

            frame = cv2.imread(self.animal_path)
            depth_map, detections = self.engine.run_inference(frame)
            img_w = frame.shape[1]

            for det in detections:
                xmin, ymin, xmax, ymax, _, _ = det
                roi = depth_map[int(ymax*0.85):int(ymax), int(xmin):int(xmax)]
                d_raw = np.percentile(roi, 95) if roi.size > 0 else 0.1

                model = self.formula_option.get()
                if model == "Linear Interpolation" or len(x_ref) < 3:
                    m_slope, c_off = np.polyfit(x_ref, 1.0/np.array(y_ref), 1)
                    z = 1.0 / (m_slope * d_raw + c_off)
                else:
                    coeffs = np.polyfit(x_ref, y_ref, 2)
                    z = coeffs[0]*(d_raw**2) + coeffs[1]*d_raw + coeffs[2]

                if z <= 0 or z > 80:
                    z = y_ref[(np.abs(np.array(x_ref) - d_raw)).argmin()]

                dist = z
                try:
                    xl, xr = float(self.ent_x_l.get()), float(self.ent_x_r.get())
                    focal = (abs(xr - xl) * 3.0) / 2.0
                    x_center = (xmin + xmax) / 2
                    x_meter = ((x_center - img_w/2) * z) / focal
                    dist = math.sqrt(z**2 + x_meter**2)
                except: pass

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
                cv2.putText(frame, f"{dist:.2f}m", (int(xmin), int(ymin)-15), 0, 1.2, (0, 255, 0), 3)

            cv2.imshow("ANALIZ SONUCU", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            messagebox.showerror("Hata", str(e))

if __name__ == "__main__":
    app = WildlifeMetricPrototype()
    app.mainloop()