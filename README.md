DL 2025 Assignment 2 — Image Captioning, Robustness, and Model Attribution

------------------------------------------------------------------------------------------------------------------------------------------

Team Members

| Name                   | Roll Number    |
|------------------------|----------------|
| Sathvik Pratapagiri    | 22CS10053      |
| VKT Sandeep            | 22CS10079      |
| Lubesh Sharma          | 22CS30065      |


------------------------------------------------------------------------------------------------------------------------------------------

Repository Structure

This submission is organized into two notebooks and one report:

| Component        | File Name                        | Description                                                                      |
|------------------|----------------------------------|----------------------------------------------------------------------------------|
| Part A & Part B  | `PART_A_and_B.ipynb`             | Implements zero-shot captioning, custom model training, and robustness tests     |
| Part C           | `PART_C.ipynb`                   | Implements BERT-based classifier to distinguish SmolVLM vs custom model captions |
| Report           | `DL_report.pdf`                  | Project methodology, results, and analysis summary                               |
| README           | `README.md`                      | Instructions to run the project and team details                                 |
                                                   


Output Folder Contents

The `output/` directory contains results generated from each part of the assignment:

From Part A & B (via `PART_A_and_B.ipynb`)
| File Name                                 | Description                                                                                |
|-------------------------------------------|--------------------------------------------------------------------------------------------|
| `output/partA_train_output.csv`	        | Custom model outputs on training data with reference captions                              |
| `output/partA_test_output.csv`            | Custom model outputs on test data with reference captions                                  |
| `output/SMOL_PARTA_outputs.csv`           | Zero-shot SmolVLM outputs with references (for test set)                                   |
| `output/partB_custom.csv`                 | Custom model captions under different occlusion levels                                     |
| `output/output_smolvlm_partb.csv`         | SmolVLM captions under different occlusion levels                                          |

From Part C (via `PART_C.ipynb`)
| File Name                                 | Description                                                                                |
|-------------------------------------------|--------------------------------------------------------------------------------------------|
| `output/part_c_metrics.csv`               | Precision, Recall, and F1 scores on the test set (macro averaged)                          |
| `output/part_c_predictions.csv`           | Row-wise predictions with inputs and predicted labels (SmolVLM or Custom)                  |
| `output/occlusion_level_performance.png`  | Plot showing classifier accuracy or performance across occlusion levels                    |
| `output/confusion_matrix.png`             | Visualized confusion matrix of classifier predictions                                      |

------------------------------------------------------------------------------------------------------------------------------------------

Assignment Overview

The goal is to design and evaluate an image captioning pipeline in three phases:

1. Part A: Captioning Models  
   - Use SmolVLM for zero-shot image captioning.  
   - Train a custom Vision Transformer (ViT) + Transformer decoder-based model.  
   - Benchmark both models using BLEU, ROUGE-L, and METEOR scores.

2. Part B: Robustness via Occlusion  
   - Apply occlusion (10%, 50%, 80%) to images by masking patches.  
   - Evaluate the impact on captioning performance.

3. Part C: Source Classifier  
   - Use a BERT-based classifier to predict whether a caption came from SmolVLM or our custom model.  
   - Input format: `<original_caption> <SEP> <generated_caption> <SEP> <perturbation_percentage>`

------------------------------------------------------------------------------------------------------------------------------------------

Setup Instructions

Python Libraries

Ensure these libraries are installed:

pip install torch torchvision transformers datasets scikit-learn numpy matplotlib Pillow evaluate

Ensure that these imports are being made:

for Part_A_and_B.ipynb: 

## import nltk
## nltk.download('wordnet')
## nltk.download('punkt')

## import os
## import numpy as np
## import pandas as pd
## import matplotlib.pyplot as plt
## from PIL import Image
## from tqdm import tqdm

## import torch
## import torch.nn as nn
## from torch.utils.data import Dataset, DataLoader
## from torchvision import transforms

## from transformers import (
##     ViTModel, GPT2LMHeadModel, ViTImageProcessor, GPT2Tokenizer,
##     AutoProcessor, AutoModelForVision2Seq
## )
## from transformers.image_utils import load_image

## import evaluate
## from evaluate import load
## from nltk.translate.meteor_score import meteor_score

## import base64

## from IPython.display import HTML

for Part_C.ipynb:

## import torch
## import torch.nn as nn
## import torch.optim as optim
## from torch.utils.data import Dataset, DataLoader
## from transformers import BertTokenizer, BertModel
## import pandas as pd
## import numpy as np
## from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
## import matplotlib.pyplot as plt
## import seaborn as sns
## from tqdm import tqdm
## import os
## import re

------------------------------------------------------------------------------------------------------------------------------------------

Hardware Requirements

- GPU support required (at least 15GB memory recommended)
- Compatible with Google Colab Free Tier (T4 GPU)

------------------------------------------------------------------------------------------------------------------------------------------

Execution Instructions

Part A — Image Captioning

File: `PART_A_and_B.ipynb`

1. Run `zero_shot_captioning()` to generate baseline captions using SmolVLM.
2. Implement and initialize `ImageCaptionModel` with:
   - ViT as encoder (ViT-Small-Patch16-224)
   - GPT2/small decoder
3. Train using `train_model()` and test using `evaluate_model()`.

Part B — Occlusion Robustness

File: `PART_A_and_B.ipynb`

1. Use `occlude_image()` to mask 10%, 50%, or 80% of image patches.
2. Evaluate both captioning models on occluded images using `evaluate_on_occluded_images()`.
3. Save original and generated captions for each perturbation level.

Part C — Caption Attribution

File: `PART_C.ipynb`

1. Create dataset using captions generated in Part B.
2. Define `CaptionClassifier`, a BERT-based binary classifier.
3. Train with `train_classifier()` and evaluate using `evaluate_classifier()` on unseen images.

------------------------------------------------------------------------------------------------------------------------------------------

Expected Outputs

- BLEU, ROUGE-L, and METEOR scores for both SmolVLM and custom models.
- Performance degradation table across occlusion levels.
- Precision, Recall, and F1 scores for the BERT classifier.

------------------------------------------------------------------------------------------------------------------------------------------

📌 Notes

- Dataset: (https://drive.google.com/file/d/1FMVcFM78XZE1KE1rIkGBpCdcdI58S1LB/view?usp=sharing)
- Evaluation metrics are calculated using HuggingFace `evaluate` or other reliable libraries.
- Ensure the naming conventions for all files are as specified in the assignment document.

------------------------------------------------------------------------------------------------------------------------------------------#   A u t o m a t i c - I m a g e - C a p t i o n i n g - a n d - R o b u s t n e s s - A n a l y s i s  
 