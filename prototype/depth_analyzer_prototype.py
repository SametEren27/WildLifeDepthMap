# type: ignore
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import threading

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_type = "DPT_Large" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def left(num):
    leftimg = cv2.imread('samples/test2.jpeg')
    left_rgb = cv2.cvtColor(leftimg, cv2.COLOR_BGR2RGB)
    input_batch = transform(left_rgb).to(device)

    with torch.no_grad():
        l_prediction = midas(input_batch)
        l_prediction = torch.nn.functional.interpolate(
            l_prediction.unsqueeze(1),
            size=left_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = l_prediction.cpu().numpy()
    plt.imshow(output, cmap='jet')
    plt.title("Depth Map Test")
    plt.show()

def right(num):

    rightimg = cv2.imread('samples/test1.jpeg')
    right_rgb = cv2.cvtColor(rightimg, cv2.COLOR_BGR2RGB)
    input_batch = transform(right_rgb).to(device)

    with torch.no_grad():
        r_prediction = midas(input_batch)
        r_prediction = torch.nn.functional.interpolate(
            r_prediction.unsqueeze(1),
            size=right_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = r_prediction.cpu().numpy()
    plt.imshow(output, cmap='jet')
    plt.title("Depth Map Test")
    plt.show()

t1 = threading.Thread(target=left, args=(...,))
t2 = threading.Thread(target=right, args=(...,))

t1.start()
t2.start()
t1.join()
t2.join()


