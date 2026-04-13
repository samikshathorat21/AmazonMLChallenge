# TheWalkingDevs — Product Price Prediction

A multimodal machine learning pipeline for predicting grocery/consumer product prices from catalog text and product images. Built in Google Colab with GPU acceleration.

---

## Overview

This notebook builds a price prediction model by combining multiple feature representations extracted from product catalog descriptions and images:

- **TF-IDF features** from product text (10,000 dimensions)
- **Semantic embeddings** via `all-MiniLM-L6-v2` (Sentence Transformers)
- **Image embeddings** via `EfficientNet-B0` (timm)
- **Hand-crafted text features** (nutrition info, certifications, category flags, packaging, etc.)

All feature matrices are horizontally stacked into a single sparse matrix and fed into a downstream regression model to predict log-transformed prices.

---

## Project Structure

```
student_resource/
├── dataset/
│   ├── train.csv           # Training data with price labels
│   └── test.csv            # Test data for submission
├── images/                 # Product images (downloaded separately)
├── tfidf_matrix.npz        # Cached TF-IDF sparse matrix
├── semantic_embeddings.npy # Cached sentence embeddings
├── image_embeddings.npy    # Cached EfficientNet image embeddings
└── precomputed_text_features.csv  # Cached hand-crafted features
```

**Output:** `test_out_updated.csv` — submission file with `sample_id` and predicted `price`.

---

## Dataset

The dataset is expected at `/content/drive/MyDrive/student_resource/dataset/` on Google Drive and must contain:

| Column | Description |
|---|---|
| `sample_id` | Unique product identifier |
| `catalog_content` | Product description text |
| `image_link` | URL or filename of product image |
| `price` | Target variable (train only) |

Combined train + test size observed: **150,000 rows × 4 columns**.

---

## Pipeline

### 1. Setup
Mounts Google Drive and configures dataset/image paths.

### 2. Data Loading
Reads `train.csv` and `test.csv`, applies `log1p` transformation to the price column, and concatenates train and test for consistent feature engineering.

### 3. Hand-Crafted Text Features
Extracts 29 features from `catalog_content` across these categories:

- **Basic stats:** content length, word count, items-per-quantity (IPQ)
- **Dietary claims:** gluten-free, non-GMO, dairy-free
- **Nutrition:** grams, protein, carbs, sugar
- **Certifications:** USDA Organic, Fair Trade, ISO, FDA approved
- **Product category:** snack, beverage, supplement, bakery, frozen
- **Packaging:** pack size, eco-friendly, container type
- **Premium signals:** premium/luxury/artisan keywords, import origin
- **Chocolate type:** chocolate, dark, milk, white
- **Brand encoding:** label-encoded brand, rare brands grouped

### 4. TF-IDF Features
Fits a `TfidfVectorizer` with up to 10,000 features on the full corpus. Results are cached to `tfidf_matrix.npz` for reuse.

### 5. Semantic Embeddings
Uses `sentence-transformers/all-MiniLM-L6-v2` to encode `catalog_content` into dense 384-dim vectors. Results are cached to `semantic_embeddings.npy`.

### 6. Image Embeddings
Uses a pretrained `EfficientNet-B0` (via `timm`, no classification head) to extract 1280-dim image feature vectors. Missing or unreadable images fall back to a zero vector. Results are cached to `image_embeddings.npy`.

### 7. Feature Matrix Assembly
Horizontally stacks all features into sparse CSR matrices:

```
X = [TF-IDF (10000) | hand-crafted (29) | semantic (384) | image (1280)]
```

### 8. Submission
Generates `test_out_updated.csv` with columns `sample_id` and `price` (exponentiated back from log-space predictions via `final_predictions`).

---

## Requirements

```
pandas
numpy
scipy
scikit-learn
sentence-transformers
timm
torch
Pillow
tqdm
google-colab          # for Drive mounting
```

Install in Colab:
```bash
pip install timm sentence-transformers
```

---

## Usage

1. Upload the notebook to Google Colab.
2. Place your dataset and images under `MyDrive/student_resource/` as described above.
3. Set runtime to **GPU** (T4 recommended).
4. Run all cells top to bottom. Cached `.npy`/`.npz` files will be reused on subsequent runs.
5. Download `test_out_updated.csv` from the Colab working directory.

---

## Notes

- The notebook caches all heavy computations (TF-IDF, embeddings) to Google Drive. If you change the dataset, delete the cached files to force recomputation.
- The variable `final_predictions` (used in the submission cell) must be defined by the model training step, which is not included in this notebook excerpt — add your regression model between steps 7 and 8.
- GPU acceleration significantly speeds up both `SentenceTransformer` encoding and `EfficientNet` inference.
