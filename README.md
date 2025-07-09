# twitter-sentiment-analysis
# Overview
This project performs sentiment analysis on tweets using the BERT (Bidirectional Encoder Representations from Transformers) model. The goal is to classify tweets as positive, negative, or neutral with high accuracy and robust generalization.

I fine-tuned a pre-trained BERT model on a labeled dataset of tweets, achieving an impressive accuracy of ~89% on the test set.

# Motivation
Social media platforms like Twitter host vast volumes of opinionated content. Identifying the sentiment behind these short, informal texts can:

Help brands monitor public opinion

Assist in political or social research

Enable automated moderation or recommendation engines

# Model Used
Model: bert-base-uncased

Library: Hugging Face Transformers (transformers)

Tokenizer: BERT's own WordPiece tokenizer

Framework: PyTorch

The model was fine-tuned using a classification head on top of the BERT encoder. It was trained to classify each tweet into one of three sentiment classes:

Positive

Neutral

Negative

# Dataset
Source: Kaggle

Preprocessing:

Removed URLs, mentions, hashtags, emojis, and HTML characters

Tokenized using BERT tokenizer

Padded and truncated sequences to uniform length (max length = 128)

Encoded labels: 0 = Negative, 1 = Neutral, 2 = Positive

# Results
Accuracy on test set: ~89%

Loss: Monitored using validation loss during training

Evaluation Metrics: Accuracy, F1 Score (Macro & Weighted), and Confusion Matrix

# Key Features
Uses pre-trained bert-base-uncased for rich language understanding

Fine-tuning on domain-specific tweet data

Efficient text preprocessing pipeline

Custom dataset loader for handling tweet formats

Implements GPU acceleration using PyTorch for faster training

# Evaluation
Evaluation metrics were obtained using the sklearn.metrics module:

Confusion Matrix

Classification Report

Macro F1-Score

Precision & Recall

# Tech Stack
Python 3.10+

PyTorch

Hugging Face Transformers

scikit-learn

pandas

matplotlib / seaborn (for visualizations)

# Future Work
Add live tweet inference using Twitter API

Deploy as a REST API using Flask or FastAPI

Experiment with domain-adapted BERT variants like BERTweet

# License
This project is licensed under the MIT License.

# Author
Ankita Saxena
National Institute of Technology, Patna
GitHub: [https://github.com/Ankita-3012]
Email: [ankitas.ug23.ec@nitp.ac.in]
