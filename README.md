# Adult Income Binary Classification

A Deep Learning project using a Multilayer Perceptron (MLP) built with TensorFlow/Keras to predict whether an individual's annual income is above or below **$50,000**, using the Adult Census Income dataset.

## Project Information

| Item | Details |
|---|---|
| Project | Deep Learning — PR 2 |
| Institute | Red & White Skill Education |
| Problem Type | Binary Classification |
| Dataset | Adult Income / Census Income Dataset |
| Framework | TensorFlow 2.x / Keras |
| Libraries | NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn |
| Target | `income` |

## Repository Structure

```text
Adult_Income_Binary_Classification/
├── Adult_Income_Binary_Classification.ipynb
├── adult.csv
├── DL_PR2.html
├── requirements.txt
├── README.md
└── plots/
    ├── eda_class_balance.png
    ├── activation_comparison.png
    ├── initialiser_convergence.png
    ├── weight_distributions.png
    ├── loss_f1_comparison.png
    ├── batchnorm_dynamics.png
    ├── optimiser_convergence.png
    ├── roc_curves.png
    └── results_table.png
```

*Add the HTML export and saved plot images after generating them in the notebook.*

## Dataset

The Adult Income dataset was extracted from the 1994 US Census database and contains demographic and employment information.

- **UCI:** https://archive.ics.uci.edu/dataset/2/adult
- **Kaggle:** https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
- **File:** `adult.csv`
- **Original size specified in the project brief:** 48,842 records and 15 columns.

### Target Variable

| Income | Encoded value | Meaning |
|---|---:|---|
| `<=50K` | 0 | Income is $50,000 or less |
| `>50K` | 1 | Income is greater than $50,000 |

The target is imbalanced (approximately 76% class 0 and 24% class 1, as described in the assignment). Therefore, accuracy alone is insufficient; class-1 precision, recall and F1-score are also evaluated.

## Project Objectives

1. Prepare the dataset for neural-network training.
2. Build a reusable MLP architecture for binary classification.
3. Compare activation functions and weight initialization techniques.
4. Evaluate loss functions for imbalanced classification.
5. Study Batch Normalization and its placement.
6. Compare optimizers and learning-rate settings.
7. Train and evaluate a final combined ANN model.

## Data Preprocessing

The notebook performs the following steps:

1. Load `adult.csv` with Pandas and inspect its shape, data types and first records.
2. Strip whitespace from all object/string columns.
3. Replace `?` with `NaN` and inspect missing values.
4. Drop rows containing missing values, as required by the assignment.
5. Drop `fnlwgt` (census sampling weight) and `education` (redundant with `education.num`).
6. Encode `income`: `<=50K` as `0` and `>50K` as `1`.
7. Apply `pd.get_dummies(drop_first=True)` to categorical input columns.
8. Apply `StandardScaler` to numeric columns only, not one-hot binary columns.
9. Split into training and testing sets using `test_size=0.2`, `random_state=42` and `stratify=y`.

The exact number of rows remaining after cleaning is printed by the notebook because it depends on the supplied CSV's missing values.

## Deep Learning Concepts and Experiments

### 1. Multilayer Perceptron (MLP)

A reusable `build_ann()` function builds a Keras Sequential network. The baseline architecture is:

```text
Input Features
      |
Dense(128, ReLU)
      |
Dense(64, ReLU)
      |
Dense(1, Sigmoid)
      |
Predicted Probability
```

A single sigmoid output neuron is used because this is binary classification. It returns a probability for the positive class (`>50K`).

### 2. Activation Functions

The hidden-layer comparison includes:

- **ReLU:** `max(0, x)`; efficient and commonly used in hidden layers.
- **Tanh:** maps values to `[-1, 1]`.
- **Sigmoid:** maps values to `(0, 1)`; hidden sigmoid units can suffer from vanishing gradients.
- **ELU:** permits negative outputs for negative inputs and can reduce the dead-neuron issue associated with ReLU.

The notebook compares training/validation accuracy and includes ReLU dead-neuron and sigmoid gradient-flow diagnostics.

### 3. Weight Initialization

The experiments compare Glorot Uniform, Glorot Normal, He Uniform, He Normal and Zeros.

- **Glorot/Xavier:** variance based on `2 / (fan_in + fan_out)`, commonly used to keep signal scales stable for tanh/sigmoid networks.
- **He:** variance based on `2 / fan_in`, designed for ReLU networks.
- **Zeros:** demonstrates the symmetry problem, where neurons in the same layer receive identical updates and cannot learn distinct representations.

### 4. Loss Functions

The notebook compares:

- **Binary Cross-Entropy (BCE):** standard loss for sigmoid-based binary classification.
- **Mean Squared Error (MSE):** comparison loss.
- **Weighted BCE:** uses class weights to give more importance to minority-class errors.
- **Focal Loss:** down-weights easy examples and focuses on harder examples.

Class-1 precision, recall and F1-score are compared to understand the effect on the minority class.

### 5. Batch Normalization

The canonical layer order tested is:

```text
Dense → BatchNormalization → ReLU
```

The notebook compares models with and without Batch Normalization, tests different BatchNorm positions, and inspects learned `gamma` and `beta` parameters.

### 6. Optimizers

The optimizer experiments include:

- Vanilla SGD
- SGD with Momentum
- RMSprop
- Adam
- Adam with explicitly configured parameters

Learning-rate sensitivity is also examined for SGD and Adam. The final configuration should be selected using the measured experiment results.

## `build_ann()` Function Parameters

| Parameter | Purpose | Default |
|---|---|---|
| `input_dim` | Number of input features | Required |
| `hidden_units` | Neurons in hidden layers | `[128, 64]` |
| `activation` | Hidden-layer activation | `relu` |
| `initializer` | Dense-layer kernel initializer | `glorot_uniform` |
| `use_batch_norm` | Add Batch Normalization | `False` |
| `optimizer` | Training optimizer | `adam` |
| `loss` | Training loss function | `binary_crossentropy` |

## Model Evaluation

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision (Class 1) | How many predicted `>50K` cases are correct |
| Recall (Class 1) | How many actual `>50K` cases are identified |
| F1-score (Class 1) | Harmonic mean of class-1 precision and recall |
| ROC-AUC | Ability to distinguish between the two income classes |

A confusion matrix, classification report and ROC curve are generated. Class-1 F1-score is especially important because the target is imbalanced.

## Experimental Results

**Replace the placeholders below with the actual metrics generated by running the notebook. No results are assumed or fabricated.**

| Model / Experiment | Test Accuracy | Precision (1) | Recall (1) | F1-score (1) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline ANN | — | — | — | — | — |
| Best Activation | — | — | — | — | — |
| Best Initializer | — | — | — | — | — |
| MSE | — | — | — | — | — |
| Weighted BCE | — | — | — | — | — |
| Focal Loss | — | — | — | — | — |
| Batch Normalization | — | — | — | — | — |
| Final Combined ANN | — | — | — | — | — |

### Results Discussion

After running the experiments, report:

- Which activation achieved the best validation performance?
- Which non-zero initializer converged fastest?
- How did Weighted BCE and Focal Loss affect class-1 recall and F1?
- Did Batch Normalization improve convergence or the train-validation gap?
- Which optimizer achieved the best class-1 F1 and converged fastest?
- Which single technique had the largest measured impact on minority-class F1?

## Visualizations

Save the generated figures in the `plots/` directory using descriptive filenames.

| Plot file | Description |
|---|---|
| `eda_class_balance.png` | Distribution of the two income classes |
| `activation_comparison.png` | Training and validation accuracy for four activations |
| `initialiser_convergence.png` | Validation accuracy for five initializers |
| `weight_distributions.png` | Initial and trained weight distributions |
| `loss_f1_comparison.png` | Class-1 precision, recall and F1 across loss strategies |
| `batchnorm_dynamics.png` | Training dynamics with and without BatchNorm |
| `optimiser_convergence.png` | Validation accuracy for optimizer experiments |
| `roc_curves.png` | ROC curves and AUC for selected models |
| `results_table.png` | Final model comparison table |

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/patelneel9080/Adult_Income_Binary_Classification.git
cd Adult_Income_Binary_Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Open `Adult_Income_Binary_Classification.ipynb`, ensure `adult.csv` is in the same directory, and run the notebook cells from top to bottom.

### 4. Export the notebook to HTML

```bash
jupyter nbconvert --to html Adult_Income_Binary_Classification.ipynb --output DL_PR2.html
```

## Requirements

Create `requirements.txt` with:

```text
tensorflow>=2.12.0
scikit-learn>=1.4.0
pandas
numpy
matplotlib
seaborn
jupyter
nbconvert
```

## Video Demonstration

- **Duration:** 5–10 minutes
- **Format:** MP4 or MOV (maximum 500 MB)
- **Recording:** Notebook screen with face visible in a picture-in-picture webcam overlay.
- **Suggested filename:** `DL_PR2_YourName_GRID.mp4`
- **Video link:** `[Add your Google Drive or unlisted YouTube URL here]`

Explain the preprocessing decisions, MLP architecture, activation functions, weight initialization, loss functions, Batch Normalization and optimizers. Add the viewable video URL here before submission.

## Suggested Git Commit History

Use meaningful commits to show project progress. For example:

```bash
git add adult.csv Adult_Income_Binary_Classification.ipynb
git commit -m "Add Adult Income dataset and initial notebook"

git add Adult_Income_Binary_Classification.ipynb
git commit -m "Add data preprocessing and EDA"

git add Adult_Income_Binary_Classification.ipynb plots/
git commit -m "Add ANN experiments and model evaluation"

git add README.md requirements.txt DL_PR2.html
git commit -m "Add project documentation and HTML export"
```

## Conclusion

This project applies a Multilayer Perceptron to the Adult Income dataset and investigates how activation functions, weight initialization, loss functions, Batch Normalization and optimizers influence binary classification.

The final model and recommended configuration must be selected using the measured validation and test results. Since the dataset is imbalanced, class-1 precision, recall and F1-score are important measures of the model's ability to identify individuals earning more than $50K.

## Author

**Neel Patel**

**GitHub Repository:**  
https://github.com/patelneel9080/Adult_Income_Binary_Classification

---

*Deep Learning — PR 2 | Red & White Skill Education*
