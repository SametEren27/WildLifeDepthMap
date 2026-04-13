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

DETECTOR_PATH = get_resource_path(os.path.join('models', 'md_v5b.0.0.pt'))

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
        except Exception as e:
            print(f">>> MOTOR HATASI: {e}")

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

    def open_calibration(self):
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

            stream = open(self.animal_path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            frame = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

            if frame is None:
                messagebox.showerror("Hata", "Resim dosyası okunamadı! Lütfen dosya isminde garip karakterler olmadığından emin olun.")
                return
            info_bar_h = int(frame.shape[0] / 18)
            analysis_frame = frame[0:frame.shape[0]-info_bar_h, :].copy()
            
            h_orig, w_orig = analysis_frame.shape[:2]

            depth_map, detections = self.engine.run_inference(analysis_frame)
            
            if len(detections) == 0:
                messagebox.showinfo("Bilgi", "Görselde hayvan algılanamadı.")
                return

            for det in detections:
                xmin, ymin, xmax, ymax, conf, cls = det
                ixmin, iymin, ixmax, iymax = int(xmin), int(ymin), int(xmax), int(ymax)

                roi = depth_map[int(iymax*0.85):iymax, ixmin:ixmax]
                d_raw = np.percentile(roi, 95) if roi.size > 0 else 0.1

                model = self.formula_option.get()
                if model == "Linear Interpolation" or len(x_ref) < 3:
                    m_slope, c_off = np.polyfit(x_ref, 1.0/np.array(y_ref), 1)
                    z = 1.0 / (m_slope * d_raw + c_off)
                else:
                    coeffs = np.polyfit(x_ref, y_ref, 2)
                    z = coeffs[0]*(d_raw**2) + coeffs[1]*d_raw + coeffs[2]

                H_CAM = 0.5    
                V_FOV = 42.0   
                
                rel_y = (iymax - (h_orig / 2)) / (h_orig / 2)
                angle_rad = math.radians(rel_y * (V_FOV / 2))
                
                if angle_rad > 0:
                    dist_geo = H_CAM / math.tan(angle_rad)
                    dist = (z * 0.9) + (dist_geo * 0.1)
                else:
                    dist = z

                try:
                    xl, xr = float(self.ent_x_l.get()), float(self.ent_x_r.get())
                    focal = (abs(xr - xl) * 3.0) / 2.0
                    x_center = (ixmin + ixmax) / 2
                    x_meter = ((x_center - w_orig/2) * dist) / focal
                    dist = math.sqrt(dist**2 + x_meter**2)
                except: pass

                cv2.rectangle(analysis_frame, (ixmin, iymin), (ixmax, iymax), (0, 255, 0), 3)
                label = f"{dist:.2f}m"
                cv2.putText(analysis_frame, label, (ixmin, iymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            img_rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            self.result_win = ctk.CTkToplevel(self)
            self.result_win.title("Analiz Sonucu")
            self.result_win.attributes("-topmost", True)
            
            disp_w = 1200
            disp_h = int((h_orig / w_orig) * disp_w)
            self.result_win.geometry(f"{disp_w}x{disp_h}")

            result_img = ctk.CTkImage(light_image=img_pil, size=(disp_w, disp_h))
            result_label = ctk.CTkLabel(self.result_win, image=result_img, text="")
            result_label.pack(fill="both", expand=True)
            result_label._image = result_img 
            
        except Exception as e:
            messagebox.showerror("Hata", f"Analiz sırasında hata: {str(e)}")

if __name__ == "__main__":
    app = WildlifeMetricPrototype()
    app.mainloop()


# H_METERS = 0.5   
# V_FOV = 42.0    

#     images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


#             img_h = frame.shape[0]
#             rel_y = (ymax - (img_h / 2)) / (img_h / 2)
#             angle_rad = math.radians(rel_y * (V_FOV / 2))
#             dist_geo = H_METERS / math.tan(angle_rad) if angle_rad > 0 else dist_ai
            
#             final_dist = (dist_ai * 0.8) + (dist_geo * 0.2)

#             base_w = 2000
#             dyn_scale = frame.shape[1] / base_w
#             dyn_thick = max(2, int(3 * dyn_scale))

#             cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
#             cv2.rectangle(frame, (int(xmin), int(ymax - roi_h)), (int(xmax), int(ymax)), (0, 0, 255), 2)

#             label_str = f"{species_name} {species_conf*100:.0f}% | {final_dist:.1f}m (Raw:{robust_depth:.1f})"
#             (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, dyn_scale, dyn_thick)
            
#             cv2.rectangle(frame, (int(xmin), int(ymin) - th - 20), (int(xmin) + tw, int(ymin)), (0, 0, 0), -1)
#             cv2.putText(frame, label_str, (int(xmin), int(ymin) - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, dyn_scale, (0, 255, 0), dyn_thick)

#         win_name = 'WildLife Detection & Depth'
#         cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) 
#         cv2.setMouseCallback(win_name, get_depth_at_click, depth_map)
        






