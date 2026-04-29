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
import logging.config
from modules.inference_engine import InferenceEngine
import seaborn
import pandas
import matplotlib.pyplot
import torchvision
import ultralytics 
import timm
import PIL

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

try:
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        torch.cuda.empty_cache() 
    else:
        DEVICE = 'cpu'
except Exception as e:
    DEVICE = 'cpu'

DETECTOR_PATH = get_resource_path(os.path.join('weights', 'md_v5b.0.0.pt'))

class WildlifeMetricPrototype(ctk.CTk):
    def __init__(self):
        super().__init__() 
        self.title("DÜ-Orman Fak. Fotokapan Proje Prototipi ")
        self.geometry("500x600")
        ctk.set_appearance_mode("light")
        
        self.configure(fg_color="#E0DB86")
        self.engine = None
        self.last_depth_map = None
        self.ref_path = None
        self.animal_path = None
        self.calib_win = None

        try:
            if os.path.exists(DETECTOR_PATH):
                self.engine = InferenceEngine(DETECTOR_PATH, DEVICE)
                print(">>> MOTOR HAZIR.")
            else:
                print(f">>> HATA: Model dosyası bulunamadı: {DETECTOR_PATH}")

        except Exception as e:
            import traceback
            print(">>> KRİTİK BAŞLATMA HATASI DETAYI:")
            traceback.print_exc()        

        self.setup_ui()

    def copy_to_clipboard(self, event=None):
        full_text = self.click_label.cget("text")
        if "---" in full_text: return  
        
        try:
            val = full_text.split(":")[1].split("|")[0].strip()
            self.clipboard_clear()
            self.clipboard_append(val)
            self.update()         

            self.click_label.configure(text_color="white")
            self.after(200, lambda: self.click_label.configure(text_color="#242820"))
        except:
            pass
            
    def setup_ui(self):
        f_frame = ctk.CTkFrame(self, fg_color="white")
        f_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(f_frame, text="Referans Seçiniz", command=self.select_ref, fg_color="#97dd36",text_color="#242820").pack(side="left", padx=10, pady=10, expand=True)
        ctk.CTkButton(f_frame, text="Analiz Edeceğiniz Görseli Seçiniz", command=self.select_animal ,fg_color="#97dd36",text_color="#242820").pack(side="left", padx=10, pady=10, expand=True)

        self.status_label = ctk.CTkLabel(self, text="Henüz Seçim Yapılmadı", font=("Arial", 14))
        self.status_label.pack(pady=10)

  
        self.roi_mode = ctk.CTkSegmentedButton(self, 
            values=["AUTO", "NEAR", "MID", "FAR"],
            command=self.update_roi_logic)
        self.roi_mode.set("AUTO")
        self.roi_mode.pack(pady=10, padx=20, fill="x")


        self.info_frame = ctk.CTkFrame(self, fg_color="#97dd36")
        self.info_frame.pack(pady=5, padx=20, fill="x")
        self.click_label = ctk.CTkLabel(self.info_frame, text="GÜNCEL AI DEĞERİ: ---", font=("Arial", 18, "bold"), text_color="#242820")
        self.click_label.pack(pady=10)

        self.click_label.bind("<Button-1>", self.copy_to_clipboard)
        self.info_frame.bind("<Button-1>", self.copy_to_clipboard)

        self.click_label.configure(cursor="hand2")
        self.info_frame.configure(cursor="hand2")

        self.ref_frame = ctk.CTkFrame(self, fg_color="white")
        self.ref_frame.pack(pady=10, padx=20, fill="both")
        self.meters = [1, 3, 5, 7, 9, 11, 13, 15]
        self.entries = {}
        for i, m in enumerate(self.meters):
            r, c = divmod(i, 2)
            row = ctk.CTkFrame(self.ref_frame, fg_color="white"); row.grid(row=r, column=c, padx=10, pady=5, sticky="ew")
            ctk.CTkLabel(row, text=f"{m}m Değeri:", width=80).pack(side="left")
            ent = ctk.CTkEntry(row, width=120); ent.pack(side="right", padx=5)
            self.entries[m] = ent

        h_frame = ctk.CTkFrame(self, fg_color="#78C012")
        h_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(h_frame, text="YATAY TELAFİ (3m Çubukları)", font=("Arial", 12, "bold")).pack(pady=2)
        h_row = ctk.CTkFrame(h_frame, fg_color="transparent")
        h_row.pack(pady=5)
        ctk.CTkLabel(h_row, text="Sol X(Pixel Değeri):").pack(side="left", padx=5)
        self.ent_x_l = ctk.CTkEntry(h_row, width=80); self.ent_x_l.pack(side="left", padx=5)
        ctk.CTkLabel(h_row, text="Sağ X(Pixel Değeri):").pack(side="left", padx=5)
        self.ent_x_r = ctk.CTkEntry(h_row, width=80); self.ent_x_r.pack(side="left", padx=5)

        self.formula_option = ctk.CTkComboBox(self, values=["Linear Interpolation", "2. Degree Polynomial"])
        self.formula_option.pack(pady=10); self.formula_option.set("Linear Interpolation")

        self.calib_btn = ctk.CTkButton(self, text="KALİBRASYON EKRANINI AÇ", command=self.open_calibration, state="disabled", fg_color="#97dd36" ,text_color="#050A01")
        self.calib_btn.pack(pady=5)
        self.run_btn = ctk.CTkButton(self, text="ANALİZİ BAŞLAT", command=self.start_analysis, fg_color="green", height=45, state="disabled")
        self.run_btn.pack(pady=20)

    def select_ref(self):
        self.ref_path = filedialog.askopenfilename()
        if self.ref_path: self.calib_btn.configure(state="normal", fg_color="Black")
        self.calib_btn.configure(state="normal", fg_color="White")
        self.status_label.configure(text="Görsel Yüklendi!", font=("Arial", 18, "bold"), text_color="#926615")


    def select_animal(self):
        self.animal_path = filedialog.askopenfilename()
        if self.animal_path: self.run_btn.configure(state="normal")


    def update_roi_logic(self, value):
        print(f">>> Manuel ROI Modu Seçildi: {value}")


    def open_calibration(self):

        if self.engine is None:
            messagebox.showerror("Hata", "Görüntü işleme motoru yüklenemedi! \nTerminaldeki hata mesajına bakın.")
            return
            
        if self.calib_win is not None and self.calib_win.winfo_exists():
            self.calib_win.focus(); return 

        stream = open(self.ref_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        frame = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

        if frame is None:
            messagebox.showerror("Hata", "Referans resmi açılamadı!")
            return
        info_bar_h = int(frame.shape[0] / 18)
        cropped_frame = frame[0:frame.shape[0]-info_bar_h, :]
        depth_map, _ = self.engine.run_inference(cropped_frame)
        self.last_depth_map = depth_map

        self.orig_h, self.orig_w = cropped_frame.shape[:2]
        display_w = 1280
        display_h = int((self.orig_h / (self.orig_w * 2)) * display_w)

        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_jet = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        combined = np.hstack((cropped_frame, depth_jet)) 
        
        img_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        self.calib_win = ctk.CTkToplevel(self)
        self.calib_win.title("Kalibrasyon Ekranı")

        self.calib_win.geometry(f"{display_w}x{display_h}")

        self.ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(display_w, display_h))
        
        self.img_label = ctk.CTkLabel(self.calib_win, image=self.ctk_img, text="" , fg_color="transparent" ,corner_radius=0,)
        self.img_label.pack(fill="both", expand=True, padx=0, pady=0)
        
  
        self.img_label.bind("<Button-1>", self.on_tkinter_click)
     

    def on_tkinter_click(self, event):
        curr_width = self.img_label.winfo_width()
        curr_height = self.img_label.winfo_height()
        
        one_img_width = curr_width / 2
        
        x = event.x
        clicked_x = x if x < one_img_width else x - one_img_width

        real_x = int((clicked_x / one_img_width) * self.orig_w)
        real_y = int((event.y / curr_height) * self.orig_h)
        
        real_x = max(0, min(real_x, self.orig_w - 1))
        real_y = max(0, min(real_y, self.orig_h - 1))
        

        d_raw = float(self.last_depth_map[real_y, real_x])
        
        self.click_label.configure(text=f"DERİNLİK HAM DEĞERİ: {d_raw:.4f} | X: {real_x}")

    def show_result_window(self, frame):
        res_win = ctk.CTkToplevel(self)
        res_win.title("Analiz Sonucu")
        
        h, w = frame.shape[:2]
        # En-boy oranını koruyarak yeni boyut hesapla
        max_w, max_h = 900, 700
        ratio = min(max_w/w, max_h/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = PIL.Image.fromarray(img_rgb)
        img_ctk = ctk.CTkImage(light_image=img_pil, size=(new_w, new_h))
        
        res_label = ctk.CTkLabel(res_win, image=img_ctk, text="")
        res_label.pack(pady=10, padx=10)
        ctk.CTkButton(res_win, text="Kapat", command=res_win.destroy).pack(pady=5)
            
    def start_analysis(self):
        try:
            x_ref, y_ref = [], []
            for m, ent in self.entries.items():
                if ent.get():
                    x_ref.append(float(ent.get()))
                    y_ref.append(float(m))
            
            if len(x_ref) < 2:
                messagebox.showwarning("Hata", "Lütfen en az 2 referans noktası girin!")
                return

            stream = open(self.animal_path, "rb")
            frame = cv2.imdecode(np.frombuffer(stream.read(), np.uint8), cv2.IMREAD_COLOR)
            info_bar_h = int(frame.shape[0] / 18)
            analysis_frame = frame[0:frame.shape[0]-info_bar_h, :].copy()
            h_orig, w_orig = analysis_frame.shape[:2]

            depth_map, detections = self.engine.run_inference(analysis_frame)
            
            if not detections:
                messagebox.showinfo("Bilgi", "Görselde hayvan tespit edilemedi.")
                return

            for det in detections:
                xmin, ymin, xmax, ymax, conf, cls = det
                ixmin, iymin, ixmax, iymax = int(xmin), int(ymin), int(xmax), int(ymax)
                box_h = iymax - iymin
                cx, cy = int((ixmin + ixmax)/2), int((iymin + iymax)/2)
                
                rough_roi = depth_map[max(0,cy-5):min(h_orig,cy+5), max(0,cx-5):min(w_orig,cx+5)]
                rough_val = np.median(rough_roi) if rough_roi.size > 0 else 0.5
                
                mode = self.roi_mode.get()
                if mode == "AUTO":
                    if rough_val > 0.6:   current_case = "NEAR"
                    elif rough_val > 0.35: current_case = "MID"
                    else:                 current_case = "FAR"
                    self.roi_mode.set(current_case) 
                else:
                    current_case = mode

                if current_case == "NEAR":
                    roi = depth_map[int(iymax-box_h*0.15):iymax, ixmin:ixmax]
                    d_final = np.percentile(roi, 90) if roi.size > 0 else rough_val
                    box_color = (0, 255, 0) 
                
                elif current_case == "MID":
                    roi = depth_map[int(iymin+box_h*0.5):int(iymin+box_h*0.8), ixmin:ixmax]
                    d_final = np.mean(roi) if roi.size > 0 else rough_val
                    box_color = (255, 165, 0)
                
                else: 
                    roi = depth_map[int(iymin+box_h*0.4):int(iymin+box_h*0.6), ixmin:ixmax]
                    d_final = np.mean(roi) if roi.size > 0 else rough_val
                    box_color = (0, 0, 255)

                img_cx, img_cy = w_orig / 2, h_orig / 2
                max_r = math.sqrt(img_cx**2 + img_cy**2)
                current_r = math.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
                factor = 1 + (0.15 * ((current_r / max_r)**2))

                d_final_corrected = d_final * factor

                formula = self.formula_option.get()
                if "Linear" in formula:
                    m_slope, c_off = np.polyfit(x_ref, 1.0/np.array(y_ref), 1)
                    dist = 1.0 / (m_slope * d_final_corrected + c_off)
                else:
                    coeffs = np.polyfit(x_ref, y_ref, 2)
                    dist = coeffs[0]*(d_final_corrected**2) + coeffs[1]*d_final_corrected + coeffs[2]

                label = f"{dist:.2f}m [{current_case}]"
                cv2.rectangle(analysis_frame, (ixmin, iymin), (ixmax, iymax), box_color, 3)
                cv2.putText(analysis_frame, label, (ixmin, iymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

            self.show_result_window(analysis_frame)

        except Exception as e:
            messagebox.showerror("Hata", f"Analiz başarısız: {str(e)}")

if __name__ == "__main__":
    app = WildlifeMetricPrototype()
    app.mainloop()


