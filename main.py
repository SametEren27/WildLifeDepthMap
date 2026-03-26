# type: ignore
import os
import cv2
import torch
import numpy as np
import math
import customtkinter as ctk
from tkinter import filedialog, messagebox
from modules.inference_engine import InferenceEngine
from modules.species_classifier import SpeciesClassifier

# --- AYARLAR ---
DETECTOR_PATH = os.path.join('models', 'md_v5b.0.0.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class WildlifeMetricPrototype(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Wildlife Metric Lab v2.0 - Stabil Sürüm")
        self.geometry("950x900")
        
        # Engine'leri Güvenli Yükle
        try:
            self.engine = InferenceEngine(DETECTOR_PATH, DEVICE)
            self.classifier = SpeciesClassifier()
        except Exception as e:
            messagebox.showerror("Hata", f"Modeller yüklenemedi: {e}")

        self.last_depth_map = None
        self.ref_path = None
        self.animal_path = None

        # --- PANEL TASARIMI ---
        self.setup_ui()

    def setup_ui(self):
        # 1. Dosya Seçimi
        f_frame = ctk.CTkFrame(self)
        f_frame.pack(pady=10, padx=20, fill="x")
        self.btn_ref = ctk.CTkButton(f_frame, text="1. REF FOTO (Çubuklu)", command=self.select_ref)
        self.btn_ref.pack(side="left", padx=10, pady=10, expand=True)
        self.btn_animal = ctk.CTkButton(f_frame, text="2. HAYVAN FOTO", command=self.select_animal)
        self.btn_animal.pack(side="left", padx=10, pady=10, expand=True)

        # 2. Anlık Tıklama Bilgisi (CRITICAL: Anında güncellenir)
        self.info_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.info_frame.pack(pady=5, padx=20, fill="x")
        self.click_label = ctk.CTkLabel(self.info_frame, text="SON TIKLANAN DEĞER: ---", 
                                        font=("Arial", 16, "bold"), text_color="#FFCC00")
        self.click_label.pack(pady=10)

        # 3. Referans Girişleri (Grid yapısı ile daha düzenli)
        self.ref_frame = ctk.CTkFrame(self)
        self.ref_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.meters = [1, 3, 5, 7, 9, 11, 13, 15]
        self.entries = {}
        
        # 2 sütunlu giriş alanı
        for i, m in enumerate(self.meters):
            r, c = divmod(i, 2)
            row = ctk.CTkFrame(self.ref_frame)
            row.grid(row=r, column=c, padx=10, pady=5, sticky="ew")
            ctk.CTkLabel(row, text=f"{m}m AI:", width=60).pack(side="left")
            ent = ctk.CTkEntry(row, width=120)
            ent.pack(side="right", padx=5)
            self.entries[m] = ent

        # 4. Yatay Girişler (3m Sol/Sağ)
        h_frame = ctk.CTkFrame(self)
        h_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(h_row := ctk.CTkFrame(h_frame), text="3m Sol X:").pack(side="left", padx=5)
        self.ent_x_l = ctk.CTkEntry(h_row, width=80); self.ent_x_l.pack(side="left", padx=5)
        ctk.CTkLabel(h_row, text="3m Sağ X:").pack(side="left", padx=5)
        self.ent_x_r = ctk.CTkEntry(h_row, width=80); self.ent_x_r.pack(side="left", padx=5)
        h_row.pack(pady=5)

        # 5. Kontrol Butonları
        self.calib_btn = ctk.CTkButton(self, text="KALİBRASYON PENCERESİNİ AÇ", command=self.open_calibration, state="disabled")
        self.calib_btn.pack(pady=5)
        
        self.formula_option = ctk.CTkComboBox(self, values=["Linear Interpolation", "2. Degree Polynomial", "Pinhole Model"])
        self.formula_option.pack(pady=10); self.formula_option.set("2. Degree Polynomial")

        self.run_btn = ctk.CTkButton(self, text="ANALİZİ BAŞLAT", command=self.start_analysis, fg_color="green", state="disabled")
        self.run_btn.pack(pady=10)

    # --- DOSYA VE GÖRÜNTÜ ---
    def select_ref(self):
        path = filedialog.askopenfilename()
        if path: 
            self.ref_path = path
            self.calib_btn.configure(state="normal", fg_color="blue")

    def select_animal(self):
        path = filedialog.askopenfilename()
        if path: 
            self.animal_path = path
            self.run_btn.configure(state="normal")

    def open_calibration(self):
        try:
            frame = cv2.imread(self.ref_path)
            depth_map, _ = self.engine.run_inference(frame)
            self.last_depth_map = depth_map
            
            # Görüntü oluştur
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_jet = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            
            # Ekranı bölme (Orijinal + Derinlik)
            h, w = frame.shape[:2]
            combined = np.hstack((cv2.resize(frame, (640, 480)), cv2.resize(depth_jet, (640, 480))))
            
            win_name = "KALIBRASYON (Tikla -> Deger Panele Gider) | Cikis: ESC"
            cv2.namedWindow(win_name)
            cv2.setMouseCallback(win_name, self.on_click, {"h": h, "w": w})
            
            while True:
                cv2.imshow(win_name, combined)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyWindow(win_name)
        except Exception as e:
            messagebox.showerror("Crash Önleyici", f"Görüntü işlenirken hata: {e}")

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Combined resimden asıl koordinata geçiş
            orig_w, orig_h = param["w"], param["h"]
            # Tıklanan nokta soldaki resim mi sağdaki mi? (640 sınırı)
            clicked_x = x if x < 640 else x - 640
            
            real_x = int(clicked_x * (orig_w / 640))
            real_y = int(y * (orig_h / 480))
            
            # Sınır kontrolü (Crash engelleme)
            real_x = max(0, min(real_x, orig_w - 1))
            real_y = max(0, min(real_y, orig_h - 1))
            
            val = self.last_depth_map[real_y, real_x]
            
            # PANELİ ANINDA GÜNCELLE
            self.click_label.configure(text=f"DERİNLİK: {val:.4f} | X PİKSEL: {real_x}")
            self.update_idletasks() # GUI'yi anında zorla güncelle

    # --- ANALİZ MANTIĞI ---
    def start_analysis(self):
        try:
            x_ref = []
            y_ref = []
            for m, ent in self.entries.items():
                if ent.get():
                    x_ref.append(float(ent.get()))
                    y_ref.append(m)

            if len(x_ref) < 2:
                messagebox.showwarning("Eksik Veri", "En az 2 referans girmelisiniz!")
                return

            frame = cv2.imread(self.animal_path)
            depth_map, detections = self.engine.run_inference(frame)

            for det in detections:
                xmin, ymin, xmax, ymax, _, _ = det
                # Alt kısımdan örnekleme (Henrich 2023 Mantığı)
                roi = depth_map[int(ymax*0.85):int(ymax), int(xmin):int(xmax)]
                robust_d = np.percentile(roi, 95) if roi.size > 0 else 1.0
                
                # Mesafe hesapla
                dist = self.calculate_math(robust_d, (xmin+xmax)/2, frame.shape[1], np.array(x_ref), np.array(y_ref))
                
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f"{dist:.2f}m", (int(xmin), int(ymin)-10), 0, 0.8, (0, 255, 0), 2)

            cv2.imshow("SONUC", frame)
            cv2.waitKey(0)
            cv2.destroyWindow("SONUC")
        except Exception as e:
            messagebox.showerror("Hata", f"Analiz sırasında bir sorun oluştu: {e}")

    def calculate_math(self, d_raw, x_p, img_w, xr, yr):
        model = self.formula_option.get()
        # 1. Dikey (Z)
        if model == "Linear Interpolation":
            m, c = np.polyfit(xr, 1.0/yr, 1)
            z = 1.0 / (m * d_raw + c)
        elif model == "2. Degree Polynomial":
            a, b, c = np.polyfit(xr, yr, 2)
            z = a*(d_raw**2) + b*d_raw + c
        else: # Pinhole basitleştirilmiş
            z = (3.0 * xr[np.where(yr==3)[0][0]]) / d_raw if 3 in yr else yr[0]

        # 2. Yatay (X) Düzeltme
        try:
            xl, xr_pix = float(self.ent_x_l.get()), float(self.ent_x_r.get())
            focal = (abs(xr_pix - xl) * 3.0) / 2.0
            x_m = ((x_p - img_w/2) * z) / focal
            return math.sqrt(z**2 + x_m**2)
        except:
            return z

if __name__ == "__main__":
    app = WildlifeMetricPrototype()
    app.mainloop()