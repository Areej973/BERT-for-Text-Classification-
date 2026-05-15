# BERT for Text Classification

This repository contains a complete pipeline for fine-tuning a DistilBERT model to classify Yelp reviews into five star-rating categories (1-5). The project utilizes the Hugging Face Transformers library and PyTorch.

## Project Overview

The goal of this project is to perform multi-class sentiment analysis. Unlike binary classification (positive/negative), this model predicts the specific star rating assigned by a user, ranging from 0 (1-star) to 4 (5-stars).

## Dataset

We use the yelp_review_full dataset from Hugging Face.

    Training Samples: Subsampled to 20,000 for computational efficiency.

    Split Strategy: 50% Train (10,000), 25% Validation (5,000), and 25% Test (5,000).

    Labels: 5 classes representing ratings from 1 to 5.

## Model Architecture

The project employs DistilBERT (distilbert-base-uncased), a distilled version of BERT that is 40% smaller and 60% faster while retaining 97% of BERT's performance.

### Key Features:

    Dynamic Padding: Uses DataCollatorWithPadding to optimize memory usage during training.

    Metric-Driven Training: Optimized using F1-Macro score to ensure balanced performance across all star ratings, accounting for potential class imbalances.

    Mixed Precision: Uses FP16 training to accelerate GPU computation.

## Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:
Bash

pip install torch transformers datasets accelerate evaluate scikit-learn matplotlib seaborn

## Project Workflow

    Preprocessing:

        Tokenization with a maximum length of 256 tokens.

        Column renaming and formatting for PyTorch compatibility.

    Fine-Tuning:

        Learning Rate: 2e-5

        Batch Size: 8 (per device)

        Epochs: 3

        Weight Decay: 0.01

    Evaluation:

        Accuracy and F1-Macro calculation.

        Confusion Matrix visualization to identify label confusion.

    Inference:

        Script included to predict the rating of custom, unseen reviews.

## Results

The model achieves stable performance across 3 epochs. The best model is automatically saved based on the highest F1-Macro score on the validation set.

## Usage

To train the model, run the script containing the Trainer API implementation. To perform inference on a single string:
Python

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("./bert-yelp-final")
tokenizer = AutoTokenizer.from_pretrained("./bert-yelp-final")

text = "The experience was wonderful, highly recommended."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"Predicted Rating: {prediction + 1} stars")

## Directory Structure

    results/: Directory for checkpoints and logs.

    bert-yelp-final/: Final saved model and tokenizer files.

    main.py: Full training and evaluation script.

## License

This project is open-source and available under the MIT License.
