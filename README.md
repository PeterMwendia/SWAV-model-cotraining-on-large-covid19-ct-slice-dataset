# SWAV-model-cotraining-on-large-covid19-ct-slice-dataset
# Semi-Supervised Learning using Co-training for COVID-19 Diagnosis

This is an implementation of a semi-supervised learning approach called Co-training, which uses a combination of labeled and unlabeled data to improve the performance of a COVID-19 diagnosis model. 
# Dataset
* source: https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset?resource=download
The dataset used for this project consists of CT scans of patients with COVID-19, non-COVID-19 lung infections, and healthy lungs. The objective is to classify the scans as COVID-19 positive or negative.

The code is divided into the following main sections:
1. Data loading and preprocessing
2. Co-training algorithm implementation
3. Model training and evaluation
4. Results analysis

## Data loading and preprocessing
The raw dataset consists of CT scans in PNG format. These images are preprocessed by resizing them to 64x64 pixels and converting them to NumPy arrays. The dataset is then split into 10% labeled data and 90% unlabeled data, with label distribution similar to the original dataset. The labeled data is further divided randomly into two disjoint subsets, A and B.

## SWAV Model
* reference: https://github.com/facebookresearch/swav
We used a pre-trained neural network, SWAV, developed by Facebook Research, as our base model for feature extraction. The model was fine-tuned on the labelled data and then used to extract features from the unlabelled data.

## Co-training algorithm implementation

The Co-training algorithm is implemented using two identical neural networks as classifiers, C1 and C2. The algorithm iteratively trains the classifiers on subsets A and B, respectively, and labels the data with high confidence predictions. The labeled data is then swapped between subsets A and B, and the process is repeated until convergence.

## Model training and evaluation
The labeled and unlabeled data are used to train a deep neural network for COVID-19 diagnosis. The model is trained on the labeled data and fine-tuned on the labeled data labeled by the Co-training algorithm. The performance of the model is evaluated on a holdout test set.

## Results analysis
The performance of the model is evaluated using various metrics such as accuracy, precision, recall, and F1 score. The results are compared with a baseline model trained only on the labeled data. The effect of the number of labeled data on the performance of the model is also analyzed.
