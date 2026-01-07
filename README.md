# Stock Forecasting: ANN vs SVM Comparative Study

A comprehensive implementation comparing Artificial Neural Networks (ANN) and Support Vector Machines (SVM) for stock price forecasting on Vietnamese VN30 stock market data.

## 📋 Project Overview

This project implements and compares multiple machine learning models for stock price prediction:
- **Dense ANN**: Feedforward neural network with multiple hidden layers
- **LSTM**: Long Short-Term Memory network for sequence-based prediction
- **SVM**: Support Vector Regression with multiple kernel options
- **SVM Ensemble**: Combination of multiple SVM models with different kernels

The implementation follows best practices from research papers on stock forecasting and includes comprehensive evaluation metrics and visualizations.

## 🗂️ Project Structure

```
Project/
├── dataset/
│   └── VN30_Dataset_2015_2026.csv          # Stock market data
├── config.py                                # Configuration settings
├── data_preprocessing.py                    # Data loading and preprocessing
├── ann_models.py                            # ANN/LSTM implementations
├── svm_models.py                            # SVM implementations
├── evaluation.py                            # Evaluation metrics and plots
├── main.py                                  # Main execution script
├── requirements.txt                         # Python dependencies
├── results/                                 # Output results (created at runtime)
├── models/                                  # Saved models (created at runtime)
└── plots/                                   # Visualizations (created at runtime)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

### Quick Test

Run a quick test to verify everything is working:
```bash
python main.py --quick-test
```

## 📊 Usage

### Train All Models

Train and compare all models on a specific stock:
```bash
python main.py --ticker ACB
```

Train all models on the first available ticker:
```bash
python main.py
```

### Train Specific Models

Train only Dense ANN:
```bash
python main.py --ticker ACB --models dense
```

Train only LSTM:
```bash
python main.py --ticker ACB --models lstm
```

Train only SVM:
```bash
python main.py --ticker ACB --models svm
```

### Advanced Options

Use grid search for SVM hyperparameter tuning (slower but better results):
```bash
python main.py --ticker ACB --grid-search
```

## 🔧 Configuration

Edit `config.py` to customize:
- Data split ratios
- Sequence length for LSTM
- Technical indicators
- ANN/LSTM architecture
- SVM parameters
- Training hyperparameters

### Key Configuration Options

```python
# Data Configuration
TRAIN_TEST_SPLIT = 0.8          # 80% training, 20% testing
SEQUENCE_LENGTH = 10            # Past days for prediction

# ANN Configuration
ANN_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'architecture': [...]        # Customize layers
}

# SVM Configuration
SVM_CONFIG = {
    'kernel': 'rbf',            # 'linear', 'poly', 'rbf', 'sigmoid'
    'C': 1.0,
    'epsilon': 0.1
}
```

## 📈 Features

### Data Preprocessing
- Automatic loading and cleaning of stock data
- Technical indicators generation:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - On-Balance Volume (OBV)
  - Price volatility
  - High-Low spread
- Data normalization using MinMaxScaler
- Sequence creation for time-series models

### Models

#### Dense ANN
- Multi-layer feedforward neural network
- Configurable architecture
- Dropout layers for regularization
- Early stopping to prevent overfitting

#### LSTM
- Recurrent neural network for sequential data
- Captures temporal dependencies
- Multiple LSTM layers with dropout
- Suitable for time-series forecasting

#### SVM
- Support Vector Regression (SVR)
- Multiple kernel options (linear, RBF, polynomial)
- Hyperparameter tuning via Grid Search
- Robust to outliers

#### SVM Ensemble
- Combines multiple SVM models
- Different kernels for diversity
- Average or weighted predictions

### Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (R-squared Score)
- **DA** (Directional Accuracy)

### Visualizations
- Prediction comparison plots
- Error distribution histograms
- Metrics comparison bar charts
- Scatter plots (actual vs predicted)
- Training history plots

## 📊 Output

After running the models, you'll find:

### Results Directory (`results/`)
- `model_comparison.csv`: Comprehensive metrics comparison

### Plots Directory (`plots/`)
- `predictions_comparison.png`: Time series predictions
- `error_distribution.png`: Error distribution histograms
- `metrics_comparison.png`: Bar charts of metrics
- `scatter_comparison.png`: Actual vs predicted scatter plots

### Models Directory (`models/`)
- Saved trained models for reuse

## 📖 Example Workflow

```python
from data_preprocessing import StockDataPreprocessor
from ann_models import DenseANN
from svm_models import SVMModel
from evaluation import ModelEvaluator

# 1. Prepare data
preprocessor = StockDataPreprocessor()
data = preprocessor.preprocess_for_ann(ticker='ACB', use_sequences=False)

# 2. Train ANN
ann = DenseANN(input_shape=(data['X_train'].shape[1],))
ann.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'])

# 3. Make predictions
y_pred = ann.predict(data['X_test'])
y_pred = preprocessor.inverse_transform_target(y_pred.flatten())
y_true = preprocessor.inverse_transform_target(data['y_test'])

# 4. Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model('Dense_ANN', y_true, y_pred)
evaluator.print_evaluation('Dense_ANN')
```

## 🎯 Model Performance Tips

1. **For better ANN/LSTM performance:**
   - Increase epochs (but watch for overfitting)
   - Experiment with different architectures
   - Adjust learning rate
   - Try different batch sizes

2. **For better SVM performance:**
   - Use grid search for hyperparameter tuning
   - Try different kernel functions
   - Adjust C and epsilon parameters
   - Consider feature scaling

3. **General tips:**
   - More training data usually helps
   - Feature engineering is crucial
   - Cross-validation for robust results
   - Ensemble methods often perform better

## 📝 Technical Indicators

The system automatically generates the following technical indicators:
- **Trend Indicators**: SMA, EMA, MACD
- **Momentum Indicators**: RSI
- **Volatility Indicators**: Bollinger Bands, Price Volatility
- **Volume Indicators**: OBV
- **Price-based Features**: Returns, High-Low spread

## 🔬 Research Paper Implementation

This project implements common architectures discussed in stock forecasting research:
- Multi-layer perceptron (MLP) architecture for ANN
- LSTM for capturing temporal patterns
- SVR with RBF kernel as a standard baseline
- Ensemble methods for improved robustness

## 📚 Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scikit-learn**: ML algorithms and preprocessing
- **tensorflow**: Deep learning models
- **matplotlib/seaborn**: Visualizations
- **ta**: Technical analysis indicators

## 🤝 Contributing

Feel free to extend this project by:
- Adding more models (e.g., GRU, Transformer)
- Implementing more technical indicators
- Adding fundamental analysis features
- Improving hyperparameter optimization
- Adding more evaluation metrics

## 📄 License

This project is for educational and research purposes.

## 🎓 References

- Paper: "A Comprehensive Comparative Study of Artificial Neural Network (ANN) and Support Vector Machines (SVM) on Stock Forecasting"
- Dataset: VN30 Vietnamese Stock Market (2015-2026)

## 💡 Next Steps & Improvements

See the end of this README for suggestions on extending the project.

---

## 🚀 Suggested Next Steps

1. **Add More Models**
   - Implement GRU (Gated Recurrent Unit)
   - Add Transformer-based models
   - Try ensemble methods combining ANN and SVM
   - Implement ARIMA for baseline comparison

2. **Feature Engineering**
   - Add fundamental analysis features (P/E ratio, etc.)
   - Include sentiment analysis from news
   - Add macroeconomic indicators
   - Feature selection/importance analysis

3. **Advanced Techniques**
   - Implement attention mechanisms
   - Add multi-step ahead forecasting
   - Try transfer learning across stocks
   - Implement online learning for real-time updates

4. **Evaluation & Analysis**
   - Add backtesting framework
   - Implement trading strategy evaluation
   - Calculate Sharpe ratio and other financial metrics
   - Add confidence intervals for predictions

5. **Optimization**
   - Implement Bayesian optimization for hyperparameters
   - Add neural architecture search (NAS)
   - Optimize for faster training
   - Add model compression techniques

6. **Deployment**
   - Create REST API for predictions
   - Build web dashboard for visualization
   - Add real-time data pipeline
   - Implement model monitoring

7. **Documentation & Testing**
   - Add unit tests
   - Create comprehensive documentation
   - Add example notebooks
   - Write technical report
