# FashionMNIST-EfficientNet-LoRA
# Fashion-MNIST Classifier — 94.2% Test Accuracy  
### EfficientNet-B0 Fine-Tuned with LoRA (PEFT)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-94.2%25-brightgreen)
![LoRA](https://img.shields.io/badge/PEFT-LoRA-green)

This repository contains a complete and reproducible pipeline for training a high-accuracy image classification model on the **Fashion-MNIST** dataset using **EfficientNet-B0** enhanced with **LoRA (Low-Rank Adaptation)** through the PEFT library.  
The model achieves **94.2% test accuracy** while training only ~1M parameters, making it highly efficient for low-resource environments.

---

## 📌 Key Features
- Fine-tuned **EfficientNet-B0** using LoRA adapters (PEFT)
- **94.2% test accuracy** on Fashion-MNIST
- Only **~1M trainable parameters**
- Clean and modular training pipeline
- Automatic checkpoint saving and Google Drive support
- Reproducible notebook with complete training code

---

## 📁 Repository Structure

| File/Folder | Description |
|------------|-------------|
| `best_full_model.pth` | Final trained full model (94.2% accuracy) |
| `lora_adapters/` | LoRA adapter weights (~8–10 MB) |
| `KashiBhai_Training_Notebook.ipynb` | Full training notebook with detailed code |
| `requirements.txt` | All required packages (torch, timm, peft, etc.) |

---

## 📊 Model Performance

- **Dataset:** Fashion-MNIST (60,000 train / 10,000 test images)
- **Architecture:** EfficientNet-B0 + LoRA
- **Test Accuracy:** 94.2%
- **Training Duration:** ~24 epochs (Google Colab T4 GPU)
- **Trainable Parameters:** ~1M (LoRA only)
- **Model Size:** ~21 MB (full model), ~8 MB (LoRA adapters)

---

## 🚀 Inference Usage

```python
import torch
import timm
from peft import PeftModel
from PIL import Image
from torchvision import transforms

classes = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Load base model
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=10)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "lora_adapters")

# Load full model weights
model.load_state_dict(torch.load("best_full_model.pth", map_location='cpu'))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Predict
img = Image.open("test.jpg").convert("L")
input_tensor = transform(img).unsqueeze(0)
pred = model(input_tensor).argmax(dim=1).item()

print(f"Prediction: {classes[pred]}")

👤 Author

Kashan Ikram
GitHub: @kashan-ikram

This project was developed independently within 7 days.
Fully original work with no pre-built templates or copied code.
