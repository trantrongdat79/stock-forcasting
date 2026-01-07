"""
Main execution script for stock forecasting project
Trains and compares ANN and SVM models on VN30 dataset
"""

import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import StockDataPreprocessor
from ann_models import DenseANN, LSTMModel
from svm_models import SVMModel, SVMEnsemble
from evaluation import ModelEvaluator
import config


def train_and_evaluate_dense_ann(data, preprocessor):
    """Train and evaluate Dense ANN model"""
    print("\n" + "="*80)
    print("TRAINING DENSE ANN MODEL")
    print("="*80)
    
    # Get data
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Build and train model
    input_shape = (X_train.shape[1],)
    model = DenseANN(input_shape)
    model.train(X_train, y_train, X_val, y_val, verbose=1)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = preprocessor.inverse_transform_target(y_pred_scaled)
    y_true = preprocessor.inverse_transform_target(y_test)
    
    return y_true, y_pred, model


def train_and_evaluate_lstm(data, preprocessor):
    """Train and evaluate LSTM model"""
    print("\n" + "="*80)
    print("TRAINING LSTM MODEL")
    print("="*80)
    
    # Get data
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Build and train model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = LSTMModel(input_shape)
    model.train(X_train, y_train, X_val, y_val, verbose=1)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = preprocessor.inverse_transform_target(y_pred_scaled)
    y_true = preprocessor.inverse_transform_target(y_test)
    
    return y_true, y_pred, model


def train_and_evaluate_svm(data, preprocessor, use_grid_search=False):
    """Train and evaluate SVM model"""
    print("\n" + "="*80)
    print("TRAINING SVM MODEL")
    print("="*80)
    
    # Get data
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Build and train model
    model = SVMModel(
        kernel=config.SVM_CONFIG['kernel'],
        C=config.SVM_CONFIG['C'],
        epsilon=config.SVM_CONFIG['epsilon'],
        gamma=config.SVM_CONFIG['gamma']
    )
    
    if use_grid_search:
        # Use smaller parameter grid for faster search
        param_grid = {
            'kernel': ['rbf', 'linear'],
            'C': [1, 10],
            'epsilon': [0.1, 0.5],
            'gamma': ['scale']
        }
        model.train_with_grid_search(X_train, y_train, X_val, y_val, 
                                     param_grid=param_grid, cv=3, verbose=1)
    else:
        model.train(X_train, y_train, verbose=1)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    y_pred = preprocessor.inverse_transform_target(y_pred_scaled)
    y_true = preprocessor.inverse_transform_target(y_test)
    
    return y_true, y_pred, model


def train_and_evaluate_svm_ensemble(data, preprocessor):
    """Train and evaluate SVM Ensemble"""
    print("\n" + "="*80)
    print("TRAINING SVM ENSEMBLE")
    print("="*80)
    
    # Get data
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Build and train ensemble
    ensemble = SVMEnsemble(kernels=['linear', 'rbf', 'poly'])
    ensemble.train(X_train, y_train, verbose=1)
    
    # Make predictions
    y_pred_scaled = ensemble.predict(X_test, method='average')
    y_pred = preprocessor.inverse_transform_target(y_pred_scaled)
    y_true = preprocessor.inverse_transform_target(y_test)
    
    return y_true, y_pred, ensemble


def main(ticker=None, models_to_train='all', use_grid_search=False):
    """
    Main function to run the complete pipeline
    
    Args:
        ticker: Stock ticker to analyze (None for first available)
        models_to_train: Which models to train ('all', 'ann', 'svm', 'lstm')
        use_grid_search: Whether to use grid search for SVM
    """
    print("\n" + "="*80)
    print("STOCK FORECASTING: ANN vs SVM COMPARISON")
    print("="*80)
    
    # Get available tickers
    df = pd.read_csv(config.DATA_PATH)
    available_tickers = df['Ticker'].unique()
    
    if ticker is None:
        ticker = available_tickers[0]
        print(f"\nNo ticker specified. Using first ticker: {ticker}")
    elif ticker not in available_tickers:
        print(f"\nTicker {ticker} not found. Available tickers: {available_tickers[:10]}")
        return
    
    print(f"\nAnalyzing stock: {ticker}")
    print(f"Models to train: {models_to_train}")
    
    # Store predictions and true values for each model
    predictions = {}
    true_values = {}
    
    # Train Dense ANN
    if models_to_train in ['all', 'ann', 'dense']:
        print("\n" + "-"*80)
        print("DENSE ANN PIPELINE")
        print("-"*80)
        
        preprocessor_ann = StockDataPreprocessor()
        ann_data = preprocessor_ann.preprocess_for_ann(ticker=ticker, use_sequences=False)
        y_true, y_pred, ann_model = train_and_evaluate_dense_ann(ann_data, preprocessor_ann)
        predictions['Dense_ANN'] = y_pred
        true_values['Dense_ANN'] = y_true
        ann_model.save_model()
    
    # Train LSTM
    if models_to_train in ['all', 'lstm']:
        print("\n" + "-"*80)
        print("LSTM PIPELINE")
        print("-"*80)
        
        preprocessor_lstm = StockDataPreprocessor()
        lstm_data = preprocessor_lstm.preprocess_for_ann(ticker=ticker, use_sequences=True)
        y_true, y_pred, lstm_model = train_and_evaluate_lstm(lstm_data, preprocessor_lstm)
        predictions['LSTM'] = y_pred
        true_values['LSTM'] = y_true
        lstm_model.save_model()
    
    # Train SVM
    if models_to_train in ['all', 'svm']:
        print("\n" + "-"*80)
        print("SVM PIPELINE")
        print("-"*80)
        
        preprocessor_svm = StockDataPreprocessor()
        svm_data = preprocessor_svm.preprocess_for_svm(ticker=ticker)
        y_true, y_pred, svm_model = train_and_evaluate_svm(svm_data, preprocessor_svm, 
                                                           use_grid_search=use_grid_search)
        predictions['SVM'] = y_pred
        true_values['SVM'] = y_true
        svm_model.save_model()
    
    # Train SVM Ensemble
    if models_to_train in ['all', 'svm_ensemble']:
        print("\n" + "-"*80)
        print("SVM ENSEMBLE PIPELINE")
        print("-"*80)
        
        preprocessor_svm_ens = StockDataPreprocessor()
        svm_ens_data = preprocessor_svm_ens.preprocess_for_svm(ticker=ticker)
        y_true, y_pred, svm_ens_model = train_and_evaluate_svm_ensemble(svm_ens_data, 
                                                                         preprocessor_svm_ens)
        predictions['SVM_Ensemble'] = y_pred
        true_values['SVM_Ensemble'] = y_true
    
    # Evaluate and compare all models
    if predictions:
        print("\n" + "="*80)
        print("MODEL EVALUATION AND COMPARISON")
        print("="*80)
        
        evaluator = ModelEvaluator()
        comparison_df = evaluator.generate_report_with_alignment(true_values, predictions, ticker=ticker)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nResults saved in:")
        print(f"  - {config.OUTPUT_DIR}/")
        print(f"  - {config.PLOT_DIR}/")
        print(f"  - {config.MODEL_DIR}/")
    else:
        print("\nNo models were trained!")


def run_quick_test():
    """Run a quick test with a subset of models"""
    print("\n" + "="*80)
    print("RUNNING QUICK TEST")
    print("="*80)
    
    # Get first ticker
    df = pd.read_csv(config.DATA_PATH)
    ticker = df['Ticker'].unique()[0]
    
    print(f"\nTesting with ticker: {ticker}")
    
    # Test Dense ANN only (fastest)
    preprocessor = StockDataPreprocessor()
    data = preprocessor.preprocess_for_ann(ticker=ticker, use_sequences=False)
    
    print(f"\nData prepared:")
    print(f"  Training samples: {len(data['X_train'])}")
    print(f"  Validation samples: {len(data['X_val'])}")
    print(f"  Test samples: {len(data['X_test'])}")
    print(f"  Features: {len(data['feature_columns'])}")
    
    # Train small ANN with few epochs for testing
    config.ANN_CONFIG['epochs'] = 10
    y_true, y_pred, model = train_and_evaluate_dense_ann(data, preprocessor)
    
    # Quick evaluation
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model('Dense_ANN_Test', y_true, y_pred)
    evaluator.print_evaluation('Dense_ANN_Test')
    
    print("\n" + "="*80)
    print("QUICK TEST COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Forecasting: ANN vs SVM')
    parser.add_argument('--ticker', type=str, default=None, 
                       help='Stock ticker to analyze')
    parser.add_argument('--models', type=str, default='all',
                       choices=['all', 'ann', 'dense', 'lstm', 'svm', 'svm_ensemble'],
                       help='Which models to train')
    parser.add_argument('--grid-search', action='store_true',
                       help='Use grid search for SVM hyperparameter tuning')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run a quick test with minimal training')
    
    args = parser.parse_args()
    
    if args.quick_test:
        run_quick_test()
    else:
        main(ticker=args.ticker, models_to_train=args.models, 
             use_grid_search=args.grid_search)
