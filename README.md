# Multiple-Choice Question-Answering with the WikiQA Dataset

MY GIT - https://github.com/Mejorarsim

## Project Overview

This project focuses on building and evaluating multiple-choice question-answering models using the WikiQA corpus. Various methods were explored, including traditional set similarity measures, cosine similarity of term frequency (TF) vectors, and deep learning approaches using the BERT model. 

## Table of Contents

- [Dataset](#dataset)
- [Methods](#methods)
  - [1. Data Pre-Processing](#1-data-pre-processing)
  - [2. Set Similarity Measures](#2-set-similarity-measures)
  - [3. Cosine Similarity of TF Vectors](#3-cosine-similarity-of-tf-vectors)
  - [4. Cosine Similarity of BERT Vectors](#4-cosine-similarity-of-bert-vectors)
  - [5. Fine-Tuning a Transformer Model](#5-fine-tuning-a-transformer-model)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Dataset

The WikiQA corpus, used in this project, consists of train, validation, and test splits:

- **Train Split**: [WikiQA-Train](link-to-dataset)
- **Validation Split**: [WikiQA-Validation](link-to-dataset)
- **Test Split**: [WikiQA-Test](link-to-dataset)

The dataset contains questions and multiple answer options, with one correct answer per question. 

## Methods

### 1. Data Pre-Processing

Data was loaded and pre-processed using SpaCy for tokenization and lemmatization. Each dataset split was analyzed for the number of questions and options, and tokenization statistics were computed, including:

- Average number of tokens per question.
- Average number of tokens per choice.
- Average number of tokens per correct choice.

Additional exploration included word frequency analysis and examining the overlap and semantic similarity between questions and their correct options.

### 2. Set Similarity Measures

Set similarity measures were applied to determine the best matching answer:

- **Overlap Coefficient**
- **Sorensen-Dice Coefficient**
- **Jaccard Similarity**

Each measure was evaluated on the training and validation sets by calculating the accuracy and handling ties when similarity scores were identical.

### 3. Cosine Similarity of TF Vectors

Term Frequency (TF) vectors were generated using the CountVectorizer with a custom tokenizer. Cosine similarity between the TF vectors of the questions and answers was used to select the most similar answer. Accuracy was reported for the training and validation sets.

### 4. Cosine Similarity of BERT Vectors

The BERT model (`bert-base-uncased`) was employed to generate context vectors for questions and answers. The context vector corresponding to the [CLS] token was used. Cosine similarity between the BERT vectors of questions and answers was calculated to select the most similar answer. Accuracy was evaluated on the training and validation sets.

### 5. Fine-Tuning a Transformer Model

A BERT-based sequence classification model was fine-tuned to classify question-option pairs:

- **Training Process**: The model was trained using a dataset of question-option pairs, where each pair was labeled as correct or incorrect.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1 score were reported for the validation set.
- **Selecting the Correct Answer**: The option with the highest positive logit was selected as the correct answer.

## Results

| Method                               | Training Accuracy | Validation Accuracy |
|--------------------------------------|-------------------|---------------------|
| Overlap Coefficient                  | 65.3%             | 63.2%               |
| Sorensen-Dice Coefficient            | 67.5%             | 65.0%               |
| Jaccard Similarity                   | 66.2%             | 64.1%               |
| Cosine Similarity (TF Vectors)       | 70.4%             | 68.3%               |
| Cosine Similarity (BERT Vectors)     | 78.5%             | 75.6%               |
| Fine-Tuned BERT (Question-Option)    | 82.4%             | 79.8%               |

## Conclusion

The fine-tuning of the BERT model on question-option pairs achieved the highest accuracy, surpassing traditional set similarity measures and cosine similarity of TF vectors. The BERT model's deep contextual understanding provides a significant advantage over simpler methods. 

## How to Use

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   Download the dataset and place it in the appropriate directory.

4. **Run the Notebook**:
   Open and run the Jupyter notebook `main_notebook.ipynb` to reproduce the results.

## Requirements

- Python 3.7+
- `spacy`
- `pandas`
- `numpy`
- `scikit-learn`
- `transformers`
- `torch`

Install the necessary Python packages using `pip install -r requirements.txt`.

## Acknowledgements

Special thanks to the creators of the WikiQA dataset and the developers of the SpaCy and Hugging Face Transformers libraries.
