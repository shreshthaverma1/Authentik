# Authentik 🔍

> AI-powered identity document authenticity verifier — detects AI-generated ID photos in real time.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-90.1%25-green?style=flat-square)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-96.4%25-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## The Problem

KYC (Know Your Customer) fraud is a growing threat in fintech and banking. With AI image generators like Stable Diffusion and Midjourney, fraudsters can generate photorealistic fake ID photos in seconds to:

- Open fraudulent bank accounts
- Bypass age verification systems
- Launder money
- Commit large-scale identity fraud

**$48 billion** was lost to identity fraud globally in 2023. Existing KYC tools verify document format and expiry dates — but none check whether the photo itself is AI-generated.

**Authentik fills that gap.**

---

## What It Does

Upload any ID photo — passport, driver's license, national ID — and Authentik will:

1. **Classify** the photo as real or AI-generated
2. **Score** the authenticity from 0–100%
3. **Assign** a risk level — Low, Medium, or High
4. **Recommend** an action — Pass, Caution, or Reject
5. **Explain** the decision using a Grad-CAM attention heatmap

---

## Demo
```
Input:   AI-generated passport photo
Output:  HIGH RISK — 94.3% AI probability
Action:  REJECT — Flag for manual review
```

---

## Model Architecture
```
Input Image (224×224×3)
        ↓
EfficientNet-B0 Backbone     ← pretrained on ImageNet
(frozen → unfrozen top layers)
        ↓
Global Average Pooling
        ↓
[1280-dim feature vector]
        ↓
Dropout(0.3) + Linear(1280→1)
        ↓
Sigmoid → Authenticity Score
```

### Training Strategy — Two Phase Transfer Learning

| Phase | Epochs | Backbone | Head | LR Backbone | LR Head |
|---|---|---|---|---|---|
| 1 | 1–4 | Frozen | Training | 0 | 3e-4 |
| 2 | 5–15 | Top layers unfrozen | Training | 1e-5 | 3e-4 |

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 90.02% |
| F1 Score | 89.94% |
| ROC-AUC | 96.41% |
| Train/Val Gap | <1% (no overfitting) |

Trained on **10,000 images** (5,000 real + 5,000 AI-generated) from the CIFAKE dataset.

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | EfficientNet-B0 (PyTorch) |
| Training | AdamW + CosineAnnealingLR |
| Explainability | Grad-CAM |
| Dataset | CIFAKE (HuggingFace) |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## Project Structure
```
authentik/
│
├── .streamlit/
│   └── config.toml          # theme settings
│
├── data/
│   └── cifake/
│       ├── train/
│       │   ├── REAL/         # 5,000 real images
│       │   └── FAKE/         # 5,000 fake images
│       └── test/
│           ├── REAL/         # 1,000 real images
│           └── FAKE/         # 2,000 fake images
│
├── src/
│   ├── dataset.py            # DataLoader + transforms
│   ├── model.py              # EfficientNet + custom head
│   ├── train.py              # training loop
│   ├── evaluate.py           # metrics + confusion matrix
│   └── gradcam.py            # Grad-CAM heatmap generator
│
├── outputs/
│   ├── best_model.pth        # saved best weights
│   └── plots/
│       └── confusion_matrix.png
│
├── config.py                 # all hyperparameters
├── prepare_data.py           # downloads + organizes dataset
├── app.py                    # Streamlit UI
└── requirements.txt
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/authentik.git
cd authentik
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Download and prepare dataset
```bash
python prepare_data.py
```

### 5. Train the model
```bash
python src/train.py
```

### 6. Evaluate
```bash
python src/evaluate.py
```

### 7. Launch the app
```bash
streamlit run app.py
```

---

## How Grad-CAM Works

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions of the image the model focused on when making its decision.
```
Real image  → model highlights natural grain and texture
Fake image  → model highlights unnaturally smooth skin,
               perfect edges, artificial lighting
```

This makes Authentik's decisions **explainable and auditable** — critical for use in financial and legal contexts.

---

## Future Improvements

- Train on more diverse generators (Midjourney, DALL-E, GAN-based)
- Add face consistency checker for batch uploads
- Fine-tune on domain-specific ID document datasets
- Deploy as a REST API for integration with KYC pipelines
- Add PDF report generation per verification

---

## License

MIT License — free to use, modify, and distribute.

---

<p align="center">
Built with PyTorch · Streamlit · Grad-CAM
</p>
