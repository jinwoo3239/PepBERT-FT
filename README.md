# PepBERT-FT: Antimicrobial Peptide MIC Prediction

## Overview

This repository provides the source code and computational workflow for predicting the minimum inhibitory concentration (MIC) of antimicrobial peptides (AMPs) against *Escherichia coli* (EC) and *Staphylococcus aureus* (SA).

The framework fine-tunes pretrained PepBERT for MIC regression using two sequence representation strategies: SOS token-based representation and mean pooling. Traditional machine learning models based on sequence-derived features are included for comparative evaluation. An ablation study is also conducted to evaluate the contributions of pretraining and encoder fine-tuning.

## Project Structure

| Directory | Description |
|---|---|
| `configs/` | Experimental settings and hyperparameter configurations |
| `data/` | Raw and processed datasets, data splits, and sequence features |
| `src/` | Python modules for data processing, model training, and evaluation |
| `models/` | Pretrained models and trained model checkpoints |
| `results/` | Training histories, predictions, and evaluation results |

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

All Jupyter notebooks should be executed from the project root directory. A CUDA-compatible GPU is recommended for PepBERT training.

## Using Trained Models

### Model Checkpoints

The fine-tuned PepBERT model checkpoints are organized in the following directory:

```text
models/pepbert/best_models/
```

The four model configurations for MIC prediction are:

| Model | Target organism | Pooling strategy |
|---|---|---|
| `EC_sos_best.pt` | *E. coli* | SOS token |
| `EC_mean_best.pt` | *E. coli* | Mean pooling |
| `SA_sos_best.pt` | *S. aureus* | SOS token |
| `SA_mean_best.pt` | *S. aureus* | Mean pooling |

### Input Format

Peptide sequences should be provided as uppercase amino acid sequences using standard one-letter amino acid codes. The maximum peptide length is 50 amino acids.

Example peptide sequences:

```python
sequences = [
    "KWKLFKKIGAVLKVL",
    "GIGKFLHSAKKFGKAFVGEIMNS"
]
```

Each sequence is tokenized using the original PepBERT tokenizer. The SOS and EOS tokens are added, and shorter sequences are padded with PAD tokens to a fixed length of 52 tokens.

### Input and Output Dimensions

| Parameter | Value |
|---|---|
| Maximum peptide length | 50 amino acids |
| Maximum token length | 52 |
| Vocabulary size | 29 |
| Input IDs | (N, 52) |
| Attention mask | (N, 52) |
| Encoder hidden size | 320 |
| Encoder output | (N, 52, 320) |
| SOS token representation | (N, 320) |
| Mean pooling representation | (N, 320) |
| Regression output | One pMIC value per sequence |

N represents the number of peptide sequences in a batch.

Both SOS token-based representation and mean pooling generate a 320-dimensional feature vector, which is passed through fully connected (FC) layers to predict a single pMIC value.

### Model Inference

To load and evaluate the saved PepBERT checkpoints, refer to `07_Model_Evaluation.ipynb`.

The inference procedure consists of the following steps:

1. Select the target organism and pooling strategy.
2. Load the original PepBERT tokenizer and encoder configuration.
3. Tokenize and pad peptide sequences to 52 tokens.
4. Generate input IDs and attention masks.
5. Initialize the regression model using the corresponding configuration.
6. Load the fine-tuned model checkpoint.
7. Predict a pMIC value for each peptide sequence.

Additional model training or hyperparameter optimization is not required when using a compatible fine-tuned checkpoint.

### Output Format

The model predicts a single pMIC value for each peptide sequence. Predicted pMIC values can be converted to MIC (μM) using the following equation:

```python
MIC = 10 ** (6 - pMIC)
```

## Dataset

MIC datasets were constructed using antimicrobial peptide activity records from DBAASP (Database of Antimicrobial Activity and Structure of Peptides).

- Target organisms: *E. coli* and *S. aureus*
- Input: Amino acid sequences
- Output: pMIC
- MIC unit: μM
- pMIC conversion: pMIC = 6 − log₁₀(MIC [μM])

The datasets were divided into training and independent test sets. Five predefined training–validation splits were generated using random seeds 42, 43, 44, 45, and 46.

## Workflow

Execute the Jupyter notebooks in the following order to reproduce the computational workflow:

| Notebook | Description |
|---|---|
| `00_Project_Setup.ipynb` | Project overview and experimental settings |
| `01_DBAASP_Dataset.ipynb` | Dataset construction and MIC preprocessing |
| `02_Data_Splitting.ipynb` | Generation of training and validation splits |
| `03_Traditional_ML.ipynb` | Sequence feature extraction |
| `04_Traditional_ML_Training.ipynb` | Traditional machine learning training and hyperparameter optimization |
| `05_PepBERT_Training_EC.ipynb` | PepBERT architecture selection and hyperparameter optimization for EC and SA |
| `06_Dataset_Analysis.ipynb` | Dataset characterization and statistical analysis |
| `07_Model_Evaluation.ipynb` | Model training, evaluation, and performance comparison |
| `08_PepBERT_Ablation.ipynb` | Ablation study |
| `09_Paper_Figures_Tables.ipynb.ipynb` | Generation of figures and tables |

## Models

### PepBERT

- SOS token-based representation
- Mean pooling-based representation

### Traditional Machine Learning

- Random Forest (RF)
- Support Vector Regression (SVR)
- Decision Tree (DT)
- K-Nearest Neighbors (KNN)

Sequence features include:

- Amino Acid Composition (AAC)
- Dipeptide Composition (DPC)
- Composition (CTDC)
- Transition (CTDT)
- Distribution (CTDD)
- Quasi-Sequence-Order (QSOrder)

Due to GitHub file size limitations, trained Random Forest models
using QSOrder features are not included in this repository.
These models can be regenerated using the provided training notebooks.

## Model Evaluation

Model performance is evaluated using the following metrics:

- Mean squared error (MSE)
- Root mean squared error (RMSE)
- Mean absolute error (MAE)
- Coefficient of determination (R²)
- Pearson correlation coefficient (PCC)
- Kendall tau correlation coefficient (KTC)

## Ablation Study

Three training strategies are compared:

- Randomly initialized PepBERT
- Frozen pretrained PepBERT
- Fine-tuned pretrained PepBERT

## License

The original source code in this repository is released under the MIT License.
See the [LICENSE](LICENSE) file for details.

Third-party software, pretrained models, and datasets remain subject to
their respective licenses and terms of use.

## Citation

Citation information will be provided upon publication.
