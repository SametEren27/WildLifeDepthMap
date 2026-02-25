import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


# Cihaz seçimi (Ekran kartı varsa CUDA, yoksa CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 15-25m hassasiyeti için DPT_Large yükleniyor [cite: 258]
model_type = "DPT_Large" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

# MiDaS için gereken dönüşüm araçları

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Görüntüyü oku (Geyik fotoğrafının adını buraya yaz)
img = cv2.imread('test2.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# Görselleştirme (Isı Haritası)
# Mavi=Yakın, Kırmızı=Uzak [cite: 143]
plt.imshow(output, cmap='jet')
plt.title("Göreceli Derinlik Haritası (15-25m Analizi Öncesi)")
plt.show()