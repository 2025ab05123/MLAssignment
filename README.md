# ML Classification Dashboard

A Streamlit-based interactive dashboard for evaluating multiple machine learning classification models on the breast cancer dataset. This project trains six different classifiers and provides a user-friendly interface to compare their performance metrics.

## Features

- **Multiple ML Models**: Compare 6 different classification algorithms:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Random Forest
  - XGBoost

- **Interactive Dashboard**: Upload test datasets and evaluate models in real-time

- **Comprehensive Metrics**: View multiple evaluation metrics including:
  - Accuracy
  - AUC (Area Under the ROC Curve)
  - Precision
  - Recall
  - F1 Score
  - Matthews Correlation Coefficient (MCC)

- **Visualizations**: Confusion matrix heatmap for detailed classification analysis

## Project Structure

```
.
├── app.py                  # Streamlit dashboard application
├── train_models.py         # Script to train and save all models
├── filegen.py             # Script to generate test dataset
├── test_data.csv          # Sample test dataset (50 samples)
├── models/                # Directory containing trained models
│   ├── logistic.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl         # StandardScaler for feature normalization
│   └── features.pkl       # Feature column names
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Dependencies

Install the required packages using pip:

```bash
pip install streamlit pandas scikit-learn joblib matplotlib seaborn xgboost
```

Or create a `requirements.txt` file with:

```
streamlit
pandas
scikit-learn
joblib
matplotlib
seaborn
xgboost
```

And install using:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Models

First, train all the classification models:

```bash
python train_models.py
```

This will:
- Load the breast cancer dataset from scikit-learn
- Train 6 different classification models
- Save all models to the `models/` directory
- Save the scaler and feature names for deployment

### 2. Generate Test Data (Optional)

Generate a sample test dataset:

```bash
python filegen.py
```

This creates a `test_data.csv` file with 50 random samples from the breast cancer dataset.

### 3. Launch the Dashboard

Run the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser (typically at `http://localhost:8501`).

### 4. Evaluate Models

1. Upload a test dataset CSV file using the file uploader
2. Select a model from the dropdown menu
3. View the evaluation metrics and confusion matrix

## Dataset

This project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** from scikit-learn, which contains:

- **Samples**: 569 instances
- **Features**: 30 numerical features computed from digitized images of breast mass
- **Target**: Binary classification (malignant vs. benign)

### Feature Information

The features describe characteristics of cell nuclei present in the images, including:
- Radius, texture, perimeter, area, smoothness
- Compactness, concavity, concave points
- Symmetry, fractal dimension

## Models Overview

| Model | Description | Best For |
|-------|-------------|----------|
| **Logistic Regression** | Linear model for binary classification | Baseline, interpretability |
| **Decision Tree** | Tree-based model with rule-based decisions | Non-linear patterns, interpretability |
| **KNN** | Instance-based learning algorithm | Local patterns, small datasets |
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem | Fast training, independence assumptions |
| **Random Forest** | Ensemble of decision trees | Robust performance, feature importance |
| **XGBoost** | Gradient boosting algorithm | High accuracy, handling complex patterns |

## Evaluation Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **AUC**: Model's ability to distinguish between classes
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **MCC**: Correlation between predicted and actual classifications (-1 to +1)

## Customization

### Using Your Own Dataset

To use a custom dataset:

1. Prepare a CSV file with the same features as the breast cancer dataset (30 features + 1 target column)
2. Ensure feature names match those used during training
3. Upload the CSV through the dashboard

### Adding New Models

To add additional models:

1. Import the model in `train_models.py`
2. Add it to the `models` dictionary
3. Update the model loading in `app.py`

## Troubleshooting

**Issue**: Models not found error  
**Solution**: Ensure you've run `train_models.py` first to create the `models/` directory

**Issue**: Feature mismatch error  
**Solution**: Verify your test dataset has the same features as the training data

**Issue**: AUC shows "N/A"  
**Solution**: Some models may not support probability predictions (check model configuration)

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to fork this project and submit pull requests for improvements or bug fixes.

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project is designed for educational purposes and demonstrates basic ML model evaluation workflows. For production use, consider adding cross-validation, hyperparameter tuning, and more robust error handling.
