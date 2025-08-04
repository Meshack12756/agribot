import torch
import joblib
from pathlib import Path

# 1. Define where your model files live
MODEL_DIR = Path("C:\Users\Admin\AgriBot\models")

# 2. Load the PyTorch models (.pt)
#    Let’s assume they are named pest_model1.pt … pest_model5.pt
pytorch_models = {}
for i in range(1, 6):
    model_path = MODEL_DIR / f"pest_model{i}.pt"
    # map_location="cuda" if you want GPU; else "cpu"
    pytorch_models[f"model{i}"] = torch.load(model_path, map_location="cpu")
    pytorch_models[f"model{i}"].eval()  # set to eval mode

# 3. Load the .pkl model (e.g. a RandomForestClassifier)
pkl_model_path = MODEL_DIR / "disease_classifier.pkl"
pkl_model = joblib.load(pkl_model_path)

# 4. Optionally wrap them in a helper class
class CropModelService:
    def __init__(self, torch_models, sklearn_model):
        self.torch_models = torch_models
        self.sklearn_model = sklearn_model

    def predict_pest(self, image_tensor, model_name="model1"):
        model = self.torch_models[model_name]
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0))  # add batch dim
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().item()

    def predict_disease(self, features):
        return self.sklearn_model.predict([features])[0]

# 5. Instantiate your service
crop_service = CropModelService(pytorch_models, pkl_model)
