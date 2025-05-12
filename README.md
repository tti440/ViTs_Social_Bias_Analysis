# 📊 Social Bias in Vision Transformers

**FAccT 2025 | T. Tsurumi & E. Beretta**  
*A Comparative Study Across Architectures and Learning Paradigms*

---

## 🧠 Overview

This repository contains the full experimental framework for the paper:

**“Social Bias in Vision Transformers: A Comparative Study Across Architectures and Learning Paradigms”**,  
accepted at **ACM FAccT 2025**, Athens, Greece.

We evaluate how various Vision Transformers encode and amplify social biases — including **gender**, **race**, and **intersectional identity** — across **ten self-supervised and supervised ViTs**. The models tested include **BEiT, ViT, MAE, Swin, MILAN, DINO**, and others.

Our pipeline combines:
- **iEAT (Image Embedding Association Test)** to quantify bias,
- **Grad-CAM** to visualize attention patterns,
- **PCA + t-SNE + DBSCAN** to cluster activation weights and highlight structural differences in feature space.

---

## 📁 Repository Structure

```bash
.
├── bias_research/
│   ├── experiment.py            # Main script to run all experiments
│   ├── gradcam_utils.py         # Grad-CAM generation logic
│   ├── imagenet_label.py        # ImageNet label ID to category map
│   ├── tsne_utils.py            # t-SNE, PCA, DBSCAN visualization functions
│   └── ...
├── ieat/
│   ├── api.py                   # Runs iEAT testing
│   ├── models.py                # Model wrappers for embedding extraction
│   └── ...
├── data/
│   └── experiments/             # Contains input images for gender & intersectional tests
│       ├── gender/
│       └── intersectional/
├── results/                     # Output CSVs, Grad-CAM .pkl files, and t-SNE/DBSCAN plots
├── .gitignore
├── README.md
└── requirements.txt             # Python dependencies
```

## 🚀 Running the Pipeline

To reproduce all experiments (iEAT, Grad-CAM, t-SNE, DBSCAN), simply run:

```bash
python bias_research/experiment.py
```

This will:
- Compute iEAT results at backbone/logits level
- Generate Grad-CAM maps and save .pkl data for further analysis
- Perform dimensionality reduction (PCA + t-SNE)
- Run DBSCAN clustering and save visual outputs

---

## 📂 Generated Data

### 📄 iEAT Results (`.csv`)
- **File**: `ieat_results_logits.csv`
- Contains numerical outputs from the **Image Embedding Association Test (iEAT)** across ViT models.

**Columns:**
- `X`, `Y`: **Categories** (e.g., `male`, `female`, `black male`, `black female`)
- `A`, `B`: **Target sets** (e.g., `science`, `liberal-arts`)
- `d`: Effect size (standardized mean difference)
- `x_ab`: Mean difference in cosine similarity between `X–A` and `X–B` pairs
- `y_ab`: Mean difference in cosine similarity between `Y–A` and `Y–B` pairs
- `p`: p-value from permutation testing
- `sig`: Significance level (`*` for p < 0.1, `**` for p < 0.05, `***` for p < 0.01)
- `n_t`, `n_a`: Number of target and attribute items

### 🧠 Grad-CAM Results (`.pkl`)
- Stored as:  
  **`{model_name}/{category}/{model_name}_{label}_{category}.pkl`**

  Example:
  ```
  vit/female/vit_labcoat_female.pkl
  swin/black-male/swin_labcoat_black-male.pkl
  ```

- Each `.pkl` contains:
  ```python
  {
    "grad": { "img1.jpg": grad_tensor, ... },
    "act":  { "img1.jpg": activation_tensor, ... },
    "grey": { "img1.jpg": grayscale_heatmap, ... }
  }
  ```

- These are used for:
  - Grad-CAM visualization
  - Activation weight extraction
  - Feature-space clustering (PCA, t-SNE, DBSCAN)

### 🖼️ Visual Outputs
- Grad-CAM collages:
  ```
  vit/female/vit_labcoat_female.jpg
  ```
- t-SNE & DBSCAN visualizations:
  ```
  labcoat_DBSCAN.png
  activation_weights_labcoat.png
  ```

---

## 📬 Contact

- **Elena Beretta** – [elena.beretta@vu.nl](mailto:elena.beretta@vu.nl)
- **Takehiro Tsurumi** – [t.tsurumi@student.vu.nl](mailto:t.tsurumi@student.vu.nl)

---
