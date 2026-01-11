# ML-Final-Project-2026
# Emotion Detection: Classical ML vs Transformers

Comparative study of emotion classification using Logistic Regression and DistilBERT on the Hugging Face emotion dataset.

## Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression + TF-IDF | 87.8% | 84.2% |
| DistilBERT | **92.9%** | **88.8%** |

## Dataset

- **Source**: [Hugging Face emotion dataset](https://huggingface.co/datasets/dair-ai/emotion)
- **Size**: ~20k samples (16k train, 2k validation, 2k test)
- **Classes**: 6 emotions (sadness, joy, love, anger, fear, surprise)

## Repository Structure
```
├── data/
│   ├── preprocessed_minimal/      # For BERT
│   └── preprocessed_extensive/    # For Logistic Regression
├── notebooks/
│   ├── 1_preprocessing.ipynb
│   ├── 2_model_experiments.ipynb
├── results/                        # Confusion matrices & plots
├── report/
│   └── final_report.pdf
└── requirements.txt
```

## Installation
```bash
pip install datasets transformers torch scikit-learn pandas numpy matplotlib seaborn
```

## Usage

1. Run `1_preprocessing.ipynb` to prepare data
2. Run `2_model_experiments.ipynb` to train models and show confusion matrices

## Methods

**Logistic Regression**
- TF-IDF features (5,000 max, unigrams + bigrams)
- Balanced class weights for imbalanced data
- Training: ~2 minutes on CPU

**DistilBERT**
- Pre-trained transformer, fine-tuned for 3 epochs
- Learning rate: 2e-5, batch size: 16
- Training: ~15 minutes on GPU (Google Colab T4)

## Key Findings

- BERT outperforms LR by 5.1% accuracy
- Both models struggle with "surprise" (only 3.6% of training data)
- BERT better handles negation and context ("not happy" → sadness)
- Transformers provide significant improvement despite added complexity

## Author

Louke Boom - Universiteit Antwerpen  
Project for Machine Learning, January 2026

## Acknowledgments

- Dataset: Hugging Face (dair-ai/emotion)
- Tools: Google Colab, Hugging Face Transformers, Claude AI
