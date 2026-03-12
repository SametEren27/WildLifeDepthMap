# type: ignore
import csv
import os

class OutputManager:
    def __init__(self, output_path="sonuclar.csv"):
        self.output_path = output_path

        if not os.path.exists(output_path):
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Dosya_Adi", "Tur", "Guven_Skoru", "Ham_Derinlik", "Metre"])

    def kaydet(self, dosya_adi, tur, guven, ham, metre):
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([dosya_adi, tur, f"%{guven*100:.2f}", f"{ham:.4f}", f"{metre:.2f}"])