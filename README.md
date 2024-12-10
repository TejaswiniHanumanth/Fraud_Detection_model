# Fraud_Detection_model

This project develops a computer vision solution to detect potential fraud during cash transactions in a retail environment, specifically focusing on transactions where an invoice was not generated or provided by the cashier.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
   - [Dataset Structure](#dataset-structure)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Evaluating the Model](#evaluating-the-model)


---

## Overview

The task involves developing a model to detect cash transactions at a retail counter, classify if the transaction involves an invoice, and flag instances where no invoice is generated. The dataset consists of videos showing these cash transactions, with some including invoices and others without.

We build the model using a **ResNet-50** architecture fine-tuned for the problem. The model is trained to classify two main labels:

1. **Cash Transaction**: Whether a cash transaction is detected.
2. **Invoice Provided**: Whether an invoice is generated after the transaction.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/retail-transaction-fraud.git
   cd retail-transaction-fraud
   ```

2. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```

3. Dataset

### Dataset Structure

The dataset consists of video frames and corresponding labels. Below is the detailed structure:


### CSV File Structure

Both `train_labels.csv` and `test_labels.csv` are structured as follows:

| Filename    | cash_transaction | invoice_provided |
|-------------|------------------|------------------|
| frame001.jpg | 1                | 0                |
| frame002.jpg | 0                | 1                |
| ...         | ...              | ...              |

- **Filename**: The name of the image frame.
- **cash_transaction**: Label indicating if a cash transaction was detected (1 for Yes, 0 for No).
- **invoice_provided**: Label indicating if an invoice was provided after the transaction (1 for Yes, 0 for No).

You should extract the frames from the videos yourself, as this step is not automated in the provided code.

---

4.## Model Architecture

The model is based on **ResNet-50**, which is fine-tuned to predict two labels for each input image frame:
1. **Cash Transaction** (Binary classification: 0 or 1)
2. **Invoice Provided** (Binary classification: 0 or 1)

The ResNet-50 architecture is modified with a new fully connected (FC) layer at the end to output predictions for these two tasks. The model includes dropout for regularization to prevent overfitting.

### Model Layers:
- **Conv Layers**: Pre-trained ResNet layers.
- **Fully Connected Layers**: Two output nodes for cash transaction and invoice detection.

---

5.## Training the Model

To train the model, use the `main.py` script. The training procedure includes:

- Loading and preprocessing the dataset (extracted frames).
- Initializing the ResNet-50 model with the custom FC layers.
- Training the model for a number of epochs.
- Saving the trained model in `models/optimized_model.pth`.

### Command to Train:
```bash
python main.py --mode train
```

6.## Evaluating the Model

After training, you can evaluate the model using a separate test dataset to validate its performance. The evaluation script uses the frames from the `test_frames` directory and labels from `test_labels.csv`.

### Command to Evaluate:
```bash
python main.py --mode validate
```

