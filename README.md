# 🩺 MediVLM: A Vision Language Model for Radiology Report Generation

**MediVLM** is a multimodal framework designed to generate clinically meaningful radiology reports directly from medical images by aligning visual and textual representations.

> **Our Paper**  
**“MediVLM: A Vision Language Model for Radiology Report Generation from Medical Images”**  
Debanjan Goswami, Ronast Subedi, Shayok Chakraborty  
*Findings of EMNLP 2025* (pp. 10287–10304)

---

## 🧠 Model Overview

MediVLM integrates region detection, multimodal alignment, and generative modeling:

- **🔍 Region Extraction**  
  Faster R-CNN (ResNet-34, MS-CXR pretrained) identifies salient anatomical regions.

- **🖼 Visual Encoding**  
  CLIP ViT-L/14 encodes top-8 patches augmented with an MLP-based positional prior over over bounding boxes `(x, y, w, h)`.

- **📝 Text Encoding**  
  ClinicalBERT processes radiology reports.

- **🔗 Alignment**  
  Contrastive learning via InfoNCE (τ = 0.07).

- **🔄 Fusion**  
  Cross-attention aligns visual and textual embeddings.

- **🧾 Generation**  
  GPT-2 decoder (last 4 blocks fine-tuned).

- **📉 Loss Function**  
  `L = λ₁ · L_CE + λ₂ · L_contrast`  
  where `λ₁ = 1`, `λ₂ = 0.7`

- **⚠️ Severity Scoring**  
  TF-IDF over curated severity terms and NLTK VADER-identified negative tokens.

- **🔁 Unsupervised Learning**  
  BioT5 generates pseudo-reports from ClinicalBERT-encoded region labels.

---

## ⚙️ Installation

```bash
# 1) create a fresh env and install PyTorch appropriate for your CUDA
python -m venv .venv && source .venv/bin/activate

# 2) install the package and dependencies
pip install -r requirements.txt
```

Optional (for the clinical-relevance metrics):

```bash
pip install radgraph          # RadGraph-F1
pip install RaTEScore         # RaTEScore (medical entity-aware semantic F1)
```

### 📦 Pretrained Weights (Auto-downloaded)
The first run downloads pretrained weights from Hugging Face:

- `openai/clip-vit-large-patch14` (~1.7 GB)
- `medicalai/ClinicalBERT` (~440 MB, fallback: `emilyalsentzer/Bio_ClinicalBERT`)
- `gpt2` (~500 MB)
- `QizhiPei/biot5-base` (~900 MB — only needed for unsupervised / pseudo-labelling)

## 📁 Repository Layout

```
MediVLM/
├── configs/
│   ├── iu_xray.yaml
│   ├── casia_cxr.yaml
│   └── mimic_cxr.yaml
├── medivlm/
│   ├── models/            # Faster R-CNN, CLIP, ClinicalBERT, fusion, GPT-2 decoder, BioT5
│   ├── data/              # IU X-Ray / CASIA-CXR / MIMIC-CXR datasets
│   ├── training/          # supervised + unsupervised trainers
│   ├── evaluation/        # BLEU/METEOR/ROUGE-L, BERTScore, RadGraph-F1, RaTEScore, severity
│   └── utils/             # config, logger, checkpoint helpers
├── scripts/
│   ├── train_supervised.py
│   ├── train_unsupervised.py
│   ├── generate_pseudo_reports.py
│   ├── evaluate.py
│   └── inference.py
└── tests/
    
```

## 📊 Dataset Format

All three datasets use the R2Gen-style JSON annotation format:

```
data/<dataset>/
├── annotations.json     # {"train": [...], "val": [...], "test": [...]}
└── images/
    └── <relative paths referenced in annotations.json>
```

Each entry looks like:

```json
{"id": "CXR1000_1",
 "image_path": ["1000_IM-0001-4001.png", "1000_IM-0001-3001.png"],
 "report": "The heart is normal in size. The lungs are clear. ..."}
```

### 📈 Dataset Statistics

| Split | IU X-Ray | MIMIC-CXR | CASIA-CXR (Pneumonia) |
| ----- | -------- | --------- | --------------------- |
| Train | 5.2K img / 2.8K rep | 369K / 222.8K | 1.8K / 1.8K |
| Val   | 0.7K / 0.4K | 3.0K / 1.8K | 0.1K / 0.1K |
| Test  | 1.5K / 0.8K | 5.2K / 3.3K | 0.1K / 0.1K |

### MS-CXR pretrained detector

The detector is Faster R-CNN with a ResNet-34 backbone pretrained on
[MS-CXR](https://physionet.org/content/ms-cxr/0.1/) (Boecking et al. 2022).
Point `detector.weights_path` in the YAML config at your local
checkpoint. If the weights are missing the module falls back to an
ImageNet-initialised ResNet-34 FPN — functional for the pipeline, but
the anatomical-region quality will degrade.

## 🏋️ Training

### Supervised

```bash
python scripts/train_supervised.py --config configs/iu_xray.yaml
python scripts/train_supervised.py --config configs/casia_cxr.yaml
python scripts/train_supervised.py --config configs/mimic_cxr.yaml
```

- Optimizer: AdamW  
- LR: 2e-5  
- Batch size: 32  
- Epochs: 30–50  (`epochs: 30` is sufficient for MIMIC-CXR, 50 for IU X-Ray and CASIA-CXR)
- Mixed precision enabled


### Unsupervised

```bash
# 1) generate pseudo-reports once (recommended for large corpora)
python scripts/generate_pseudo_reports.py \
    --config configs/iu_xray.yaml \
    --split train \
    --output outputs/iu_xray/pseudo.json

# 2) fine-tune MediVLM on the pseudo-reports
python scripts/train_unsupervised.py \
    --config configs/iu_xray.yaml \
    --pseudo-reports outputs/iu_xray/pseudo.json
```

The pseudo-report generator encodes Faster R-CNN region labels with
ClinicalBERT, decodes them with BioT5, and assembles one sentence per
region into a coherent pseudo-report.

---
## 📏 Evaluation

```bash
python scripts/evaluate.py \
    --config configs/iu_xray.yaml \
    --checkpoint outputs/iu_xray/best.ckpt \
    --metrics nlg bertscore radgraph ratescore severity
```

### Metrics
Metrics computed (defaults = `nlg`, `bertscore`, `severity`):

- **NLG**: BLEU-1..4, METEOR, ROUGE-L
- **BERTScore** (Zhang et al. 2019)
- **RadGraph-F1** (Yu et al. 2023) — requires `pip install radgraph`
- **RaTEScore** (Zhao et al. 2024) — requires `pip install RaTEScore`
- **Severity**: mean TF-IDF severity across the generated reports 

## Single-image inference

```bash
python scripts/inference.py \
    --config configs/iu_xray.yaml \
    --checkpoint outputs/iu_xray/best.ckpt \
    --image path/to/cxr.png \
    --severity-corpus data/iu_xray/train_reports.txt
```

Produces JSON:

```json
{
  "report": "The heart is normal in size. The lungs are clear ...",
  "severity": 0.18
}
```

## ⚡ Key Design Choices
- **Top-p patches, p=8**. values are
  `{6, 8, 12, 16}`. Override via `detector.top_p_patches`.
- **Max tokens = 77 (~4 sentences)**. 
- **Loss weights**. `λ₁ = 1.0`, `λ₂ = 0.7`
- **Temperature**. `τ = 0.07` 
- **Positional encoding of bboxes**. An MLP over the normalised
  `(x, y, w, h)` coordinates produces the position embedding that is
  added to the CLIP patch feature — replacing CLIP's own spatial prior.
- **Cross-attention fusion**. Visual tokens act as the query, text
  tokens as key/value (`text = None` at inference makes the module fall
  back to self-attention over visual tokens).
- **Fine-tuned GPT-2 blocks**. Only the last 4 transformer blocks + the
  LM head + `ln_f` are trainable.

## Ablations

| Ablation | Change |
| -------- | ------ |
| `-ClinicalBERT` | swap `text_encoder.model_name: openai/clip-vit-large-patch14` |
| `-Selective Patching` | set `detector.top_p_patches: 1` and patch the whole 224×224 image instead of the bbox crop |

## 📚 Citation

```bibtex
@inproceedings{goswami2025medivlm,
  title     = {{MediVLM}: A Vision Language Model for Radiology Report Generation from Medical Images},
  author    = {Goswami, Debanjan and Subedi, Ronast and Chakraborty, Shayok},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year      = {2025},
  pages     = {10287--10304}
}
```

## 📜 License

Released under the MIT License. Models and datasets retain their
respective licences — see their upstream sources for details.
