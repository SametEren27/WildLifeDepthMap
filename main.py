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
import threading
from modules.batchprocessor import BatchProcessor

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

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class WildlifeMetricPrototype(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.colors = {
            "bg_main": "#FFFDD0",
            "bg_sidebar": "#F0EAD6",
            "primary": "#23D24F",
            "primary_hover": "#23D24F",
            "accent": "#141A1F",
            "white": "#FFFFFF",
            "border": "#D1D1B0"
        }
        
        self.lang = "TR"
        self.TEXTS = {
            "TR": {
                "title": "DÜ-Orman Fak. Yaban Hayatı Analizi",
                "sidebar_head": "KONTROL PANELİ",
                "btn_ref": "1. Referans Seç",
                "btn_calib": "2. Kalibrasyonu Başlat",
                "btn_animal": "3. Analiz Görseli Seç",
                "settings": "Analiz Ayarları",
                "batch": "Yığın İşleme (Batch)",
                "conf": "Güven Haritası",
                "start": "ANALİZİ BAŞLAT",
                "ai_val": "GÜNCEL AI DEĞERİ",
                "dist_table": "Mesafe Kalibrasyon Tablosu",
                "enhancement": "Görüntü İyileştirme",
                "apply_filter": "Filtre Uygula",
                "filter_type": "Filtre Tipi",
                "sigma_s": "Uzaysal Hassasiyet",
                "sigma_r": "Renk Hassasiyeti",
                "none": "Yok"
            },
            "EN": {
                "title": "DU-Forestry Wildlife Analysis",
                "sidebar_head": "CONTROL PANEL",
                "btn_ref": "1. Select Reference",
                "btn_calib": "2. Start Calibration",
                "btn_animal": "3. Select Target Image",
                "settings": "Analysis Settings",
                "batch": "Batch Processing",
                "conf": "Confidence Map",
                "start": "RUN ANALYSIS",
                "ai_val": "CURRENT AI VALUE",
                "dist_table": "Distance Calibration Table",
                "enhancement": "Image Enhancement",
                "apply_filter": "Apply Filter",
                "filter_type": "Filter Type",
                "sigma_s": "Spatial Sigma",
                "sigma_r": "Range Sigma",
                "none": "None"
            }
        }

        self.engine = None
        self.last_depth_map = None
        self.ref_path = None
        self.animal_path = None
        self.calib_win = None  
        self.meters = [1, 3, 5, 7, 9, 11, 13, 15]
        self.entries = {} 
        self.roi_mode = ctk.StringVar(value="AUTO")
        self.formula_option = ctk.StringVar(value="Linear")

        self.is_batch = ctk.BooleanVar(value=False)
        self.show_conf = ctk.BooleanVar(value=False)

        self.title(self.TEXTS[self.lang]["title"])
        self.geometry("1000x750")
        ctk.set_appearance_mode("light")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkScrollableFrame(self, width=280, fg_color=self.colors["bg_sidebar"], corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.main_view = ctk.CTkFrame(self, fg_color=self.colors["bg_main"], corner_radius=0)
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.setup_widgets()
        self.setup_status_bar()
        self.refresh_ui_text() 
        
        self.init_engine()

    def setup_widgets(self):

        self.lbl_sidebar_head = ctk.CTkLabel(self.sidebar, text="", font=("Inter", 22, "bold"), text_color=self.colors["accent"])
        self.lbl_sidebar_head.pack(pady=(30, 20))

        self.btn_ref = self.create_styled_btn(self.sidebar, "btn_ref", self.select_ref)
        self.calib_btn = self.create_styled_btn(self.sidebar, "btn_calib", self.open_calibration, state="disabled")
        self.btn_animal = self.create_styled_btn(self.sidebar, "btn_animal", self.select_animal)

        self.lbl_settings = ctk.CTkLabel(self.sidebar, text="", font=("Inter", 14, "bold"), text_color=self.colors["accent"])
        self.lbl_settings.pack(pady=(40, 10))
        
        self.check_batch = ctk.CTkCheckBox(self.sidebar, text="", variable=self.is_batch, text_color=self.colors["accent"], border_color=self.colors["primary"])
        self.check_batch.pack(pady=10, padx=30, anchor="w")
        
        self.lang_toggle = ctk.CTkSegmentedButton(self.sidebar, values=["TR", "EN"], command=self.change_language, selected_color=self.colors["primary"])
        self.lang_toggle.pack(side="bottom", pady=20, padx=20, fill="x")
        self.lang_toggle.set(self.lang)

        self.run_btn = ctk.CTkButton(self.sidebar, text="", command=self.start_analysis_thread,
                                     fg_color=self.colors["primary"], hover_color=self.colors["primary_hover"],
                                     height=55, corner_radius=18, font=("Inter", 16, "bold"), state="disabled")
        self.run_btn.pack(side="bottom", pady=(10, 20), padx=20, fill="x")

        self.info_card = ctk.CTkFrame(self.main_view, fg_color=self.colors["white"], corner_radius=25, border_width=1, border_color=self.colors["border"])
        self.info_card.pack(fill="x", pady=(0, 30), padx=10)
        
        self.lbl_ai_val = ctk.CTkLabel(self.info_card, text="", font=("Inter", 20, "bold"), text_color=self.colors["primary"])
        self.lbl_ai_val.pack(pady=20)
        self.lbl_ai_val.configure(cursor="hand2")

        self.lbl_dist_table = ctk.CTkLabel(self.main_view, text="", font=("Inter", 16, "bold"), text_color=self.colors["accent"])
        self.lbl_dist_table.pack(anchor="w", padx=20, pady=(10, 5))

        self.grid_container = ctk.CTkFrame(self.main_view, fg_color="transparent")
        self.grid_container.pack(fill="x", padx=10)

        for i, m in enumerate(self.meters):
            r, c = divmod(i, 2)
            cell = ctk.CTkFrame(self.grid_container, fg_color=self.colors["white"], corner_radius=15, border_width=1, border_color=self.colors["border"])
            cell.grid(row=r, column=c, padx=12, pady=10, sticky="ew")
            self.grid_container.grid_columnconfigure(c, weight=1)
            
            ctk.CTkLabel(cell, text=f"{m}m:", font=("Inter", 14, "bold"), text_color=self.colors["accent"]).pack(side="left", padx=15)
            ent = ctk.CTkEntry(cell, placeholder_text="0.0000", border_width=0, fg_color="transparent", width=100)
            ent.pack(side="right", fill="x", expand=True, padx=10)
            self.entries[m] = ent

        self.lbl_enhancement = ctk.CTkLabel(self.sidebar, text="", font=("Inter", 14, "bold"), text_color=self.colors["accent"])
        self.lbl_enhancement.pack(pady=(25, 5)) 

        self.filter_switch = ctk.CTkSwitch(self.sidebar, text="", progress_color=self.colors["primary"])
        self.filter_switch.pack(pady=5, padx=30, anchor="w")

        self.filter_options = ctk.CTkComboBox(self.sidebar, values=["Bilateral Filter", "Edge Detection", "Gray Scale"], 
                                            fg_color=self.colors["white"], border_color=self.colors["border"],
                                            button_color=self.colors["primary"], button_hover_color=self.colors["primary_hover"])
        self.filter_options.set("Bilateral Filter")
        self.filter_options.pack(pady=10, padx=20, fill="x")

        self.lbl_s_val = ctk.CTkLabel(self.sidebar, text="", font=("Inter", 11))
        self.lbl_s_val.pack(pady=(5, 0))

        self.lbl_r_val = ctk.CTkLabel(self.sidebar, text="", font=("Inter", 11))
        self.lbl_r_val.pack(pady=(5, 0))

    def set_status(self, message, loading=False):
        try:
            self.status_bar.configure(text=f"● {message}")
            if loading:
                self.progress.start()
            else:
                self.progress.stop()
        except Exception as e:
            print(f"Status Bar Güncellenemedi: {e}")

    def generate_confidence_map(depth_map):

        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        conf_map = 1.0 - (gradient_mag / np.max(gradient_mag))
        
        conf_visual = cv2.applyColorMap((conf_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return conf_map, conf_visual

    def setup_status_bar(self):
        self.status_bar = ctk.CTkLabel(self, text="● Sistem Hazır", font=("Inter", 12), fg_color=self.colors["bg_sidebar"], text_color="#7F8C8D", anchor="w")
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10)
        self.progress = ctk.CTkProgressBar(self, mode="indeterminate", height=2, progress_color=self.colors["primary"])
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.progress.stop()

    def create_styled_btn(self, master, text_key, command, state="normal"):
        btn = ctk.CTkButton(master, text="", command=command, state=state,
                            fg_color=self.colors["white"], text_color=self.colors["accent"],
                            border_width=2, border_color=self.colors["primary"],
                            hover_color="#E8F5E9", corner_radius=15, height=48)
        btn.pack(pady=10, padx=25, fill="x")
        btn._text_key = text_key 
        return btn
    
    def setup_filter_section(self):
        t = self.TEXTS[self.lang]
        
        ctk.CTkLabel(self.sidebar, text=t["enhancement"], font=("Inter", 14, "bold"), text_color=self.colors["accent"]).pack(pady=(20, 5))

        self.filter_switch = ctk.CTkSwitch(self.sidebar, text=t["apply_filter"], progress_color=self.colors["primary"])
        self.filter_switch.pack(pady=10, padx=30, anchor="w")

        self.filter_options = ctk.CTkComboBox(self.sidebar, values=["Bilateral Filter", "Gray Scale", "Edge Detection"], 
                                             fg_color=self.colors["white"], button_color=self.colors["primary"])
        self.filter_options.set("Bilateral Filter")
        self.filter_options.pack(pady=10, padx=20, fill="x")

    def refresh_ui_text(self):
        t = self.TEXTS[self.lang]
        self.title(t["title"])
        self.lbl_sidebar_head.configure(text=t["sidebar_head"])
        self.lbl_settings.configure(text=t["settings"])
        self.lbl_ai_val.configure(text=f"{t['ai_val']}: ---")
        self.lbl_dist_table.configure(text=t["dist_table"])
        self.run_btn.configure(text=t["start"])
        self.check_batch.configure(text=t["batch"])
        for btn in [self.btn_ref, self.calib_btn, self.btn_animal]:
            btn.configure(text=t[btn._text_key])
        self.lbl_enhancement.configure(text=t["enhancement"])
        self.filter_switch.configure(text=t["apply_filter"])

    def start_analysis_thread(self):
        if self.is_batch.get():
            threading.Thread(target=self.run_batch_logic, daemon=True).start()
        else:
            threading.Thread(target=self.start_analysis, daemon=True).start()

    def start_analysis_trigger(self):

        if self.check_batch.get():
            if not os.path.isdir(self.animal_path):
                messagebox.showerror("Hata", "Batch modu için lütfen bir KLASÖR seçin!")
                return
            threading.Thread(target=self.run_batch_logic, daemon=True).start()
        else:
            self.start_analysis()

    def run_batch_logic(self):           
        if not self.animal_path or not os.path.isdir(self.animal_path):
            self.after(0, lambda: messagebox.showwarning("Hata", "Lütfen bir klasör seçin!"))
            return

        filter_settings = {
            'active': self.filter_switch.get(),
            'type': self.filter_options.get()
        }
        calib_data = {m: ent.get() for m, ent in self.entries.items() if ent.get().strip()}
    
        processor = BatchProcessor(self.engine)
        
        def update_progress(val, msg):
            self.progress.set(val)
            self.set_status(msg)

        try:
            output_path = processor.process_folder(
                self.animal_path, 
                calib_data, 
                filter_settings, 
                update_progress
            )
            self.after(0, lambda: messagebox.showinfo("Başarılı", f"İşlem tamamlandı.\nSonuçlar: {output_path}"))
            self.set_status("Toplu İşlem Tamamlandı.")
        except Exception as e:
            self.set_status(f"Batch Hatası: {str(e)}")

    def change_language(self, choice):
        self.lang = choice
        self.refresh_ui_text()

    def init_engine(self):
        def load():
            try:
                self.engine = InferenceEngine(DETECTOR_PATH, DEVICE)
                self.set_status("AI Motoru Hazır.")
            except: 
                self.set_status("Kritik Hata: AI Motoru yüklenemedi.")
        threading.Thread(target=load).start()

    def copy_to_clipboard(self, event=None):
        try:
            full_text = self.lbl_ai_val.cget("text")
            
            if "---" in full_text or ":" not in full_text: 
                return  
            
            val = full_text.split(":")[1].split("|")[0].strip()
            
            self.clipboard_clear()
            self.clipboard_append(val)
            self.update() 

            original_color = self.lbl_ai_val.cget("text_color")
            self.lbl_ai_val.configure(text_color="white") 
            self.after(200, lambda: self.lbl_ai_val.configure(text_color=original_color))
            
            self.set_status(f"Kopyalandı: {val}")
        except Exception as e:
            print(f"Kopyalama hatası: {e}")

    def select_ref(self):
        self.ref_path = filedialog.askopenfilename()
        if self.ref_path: self.calib_btn.configure(state="normal", fg_color="Black")
        self.calib_btn.configure(state="normal", fg_color="White")
        self.set_status("Görsel Yüklendi!")

    def select_animal(self):

        is_batch = self.check_batch.get() 

        if is_batch:
            path = filedialog.askdirectory(title="Analiz Edilecek Klasörü Seçin")
        else:
            path = filedialog.askopenfilename(
                title="Analiz Edilecek Görseli Seçin",
                filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png")]
            )

        if path:
            self.animal_path = path
            self.run_btn.configure(state="normal")
            
            secim_tipi = "Klasör" if is_batch else "Dosya"
            self.set_status(f"{secim_tipi} Seçildi: {os.path.basename(path)}")

    def update_roi_logic(self, value):
        print(f">>> Manuel ROI Modu Seçildi: {value}")

    def open_calibration(self):
        self.set_status("Görsel yükleniyor ve Depth Map oluşturuluyor...", loading=True)

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

    def finish_inference_ui(self):
        self.set_status("Analiz tamamlandı!", loading=False)


    def on_tkinter_click(self, event):

        if self.last_depth_map is None: return
        
        curr_w, curr_h = self.img_label.winfo_width(), self.img_label.winfo_height()
        one_img_w = curr_w / 2
        
        x = event.x
        clicked_x = x if x < one_img_w else x - one_img_w

        real_x = int((clicked_x / one_img_w) * self.orig_w)
        real_y = int((event.y / curr_h) * self.orig_h)
        
        real_x = max(0, min(real_x, self.orig_w - 1))
        real_y = max(0, min(real_y, self.orig_h - 1))

        d_raw = float(self.last_depth_map[real_y, real_x])
        
        roi = self.last_depth_map[max(0, real_y-2):min(self.orig_h, real_y+3), 
                                 max(0, real_x-2):min(self.orig_w, real_x+3)]

        std_dev = np.std(roi) if roi.size > 0 else 0
        conf_score = max(0, min(100, 100 - (std_dev * 500))) 

        t = self.TEXTS[self.lang]
        
        color = "#2ecc71" if conf_score > 85 else "#f1c40f" if conf_score > 60 else "#e74c3c"
        
        self.lbl_ai_val.configure(
            text=f"{t['ai_val']}: {d_raw:.4f} | Güven: %{conf_score:.1f} | X: {real_x}",
            text_color=color
        )

        self.copy_to_clipboard()

    def show_result_window(self, frame):
        res_win = ctk.CTkToplevel(self)
        res_win.title("Analiz Sonucu")
        
        h, w = frame.shape[:2]
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
            self.set_status("Analiz yapılıyor...", loading=True)
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

                sorted_indices = np.argsort(x_ref)
                x_sorted = np.array(x_ref)[sorted_indices]
                y_sorted = np.array(y_ref)[sorted_indices]

                rough_dist = np.interp(rough_val, x_sorted, y_sorted)

                mode = self.roi_mode.get()
                if mode == "AUTO":
                    if rough_dist < 5.0:      current_case = "NEAR"
                    elif rough_dist < 15.0:   current_case = "MID"
                    else:                     current_case = "FAR"
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
                # factor = 1 + (0.15 * ((current_r / max_r)**2))
                factor = 1.0 
                d_final_corrected = d_final * factor
                dist = np.interp(d_final_corrected, x_sorted, y_sorted)

                formula = self.formula_option.get()
                if "Linear" in formula:
                    sorted_indices = np.argsort(x_ref)
                    x_sorted = np.array(x_ref)[sorted_indices]
                    y_sorted = np.array(y_ref)[sorted_indices]
                        
                    dist = np.interp(d_final_corrected, x_sorted, y_sorted)
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

## knk range kısmının 2 asamalı calısması gerektigidnen emin ol ic ice calısıcaklar once default olcak sonra near mid far yapıcak ve formulleri tekrar duzenle 