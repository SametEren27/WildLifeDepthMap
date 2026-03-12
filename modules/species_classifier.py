# import ultralytics
# import inference_engine 
# import torch

# classifier = torch.hub.load('ultralytics/YOLO11m-cls', 'custom', path=classifier_path)

# def classify_species(capture):

#     capture = inference_engine.run_inference(detection)
#     classname = capture.classname
#     confidencescore = capture.confidincescore

#     return classname, confidencescore

from ultralytics import YOLO

class SpeciesClassifier:
    def __init__(self, model_path='yolo11m-cls.pt'):
        # This will download the model (~10MB) automatically on the first run
        self.model = YOLO(model_path)

    def predict(self, animal_crop):
        # We perform inference on the small cropped image
        results = self.model(animal_crop, verbose=False)
        
        # Extract the best prediction
        result = results[0]
        class_id = result.probs.top1
        class_name = result.names[class_id]
        confidence = result.probs.top1conf.item()
        
        return class_name, confidence