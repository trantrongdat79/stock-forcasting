"""
Configuration file for stock forecasting project
Based on ANN and SVM comparative study
"""

# Data Configuration
DATA_PATH = 'dataset/VN30_Dataset_2015_2026.csv'
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing
VALIDATION_SPLIT = 0.2  # 20% of training data for validation
SEQUENCE_LENGTH = 10  # Number of past days to use for prediction

# Feature Engineering
TECHNICAL_INDICATORS = {
    'SMA': [5, 10, 20],  # Simple Moving Average windows
    'EMA': [5, 10, 20],  # Exponential Moving Average windows
    'RSI': 14,           # Relative Strength Index period
    'MACD': True,        # Moving Average Convergence Divergence
    'BB': 20,            # Bollinger Bands period
    'OBV': True,         # On-Balance Volume
}

# ANN Configuration (Based on typical architecture in research papers)
ANN_CONFIG = {
    'architecture': [
        {'type': 'dense', 'units': 64, 'activation': 'relu'},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'dense', 'units': 32, 'activation': 'relu'},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'dense', 'units': 16, 'activation': 'relu'},
        {'type': 'dense', 'units': 1, 'activation': 'linear'}
    ],
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae', 'mse'],
    'epochs': 100,
    'batch_size': 32,
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 10,
        'restore_best_weights': True
    }
}

# LSTM Configuration (Deep Learning variant)
LSTM_CONFIG = {
    'architecture': [
        {'type': 'lstm', 'units': 50, 'return_sequences': True},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'lstm', 'units': 50, 'return_sequences': False},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'dense', 'units': 25, 'activation': 'relu'},
        {'type': 'dense', 'units': 1, 'activation': 'linear'}
    ],
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae', 'mse'],
    'epochs': 100,
    'batch_size': 32,
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 10,
        'restore_best_weights': True
    }
}

# SVM Configuration (Common configurations from research)
SVM_CONFIG = {
    'kernel': 'rbf',  # Options: 'linear', 'poly', 'rbf', 'sigmoid'
    'C': 1.0,         # Regularization parameter
    'epsilon': 0.1,   # Epsilon in epsilon-SVR model
    'gamma': 'scale', # Kernel coefficient
    'max_iter': 1000
}

# Grid Search for SVM hyperparameter tuning
SVM_GRID_SEARCH = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto']
}

# Evaluation Metrics
METRICS = [
    'MAE',   # Mean Absolute Error
    'MSE',   # Mean Squared Error
    'RMSE',  # Root Mean Squared Error
    'MAPE',  # Mean Absolute Percentage Error
    'R2',    # R-squared Score
    'DA',    # Directional Accuracy
]

# Output Configuration
OUTPUT_DIR = 'results'
MODEL_DIR = 'models'
PLOT_DIR = 'plots'
