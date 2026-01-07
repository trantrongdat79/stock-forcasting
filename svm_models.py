"""
Support Vector Machine (SVM) models for stock forecasting
Includes SVR with different kernels and hyperparameter tuning
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import joblib
import os
import config


class SVMModel:
    """SVM model for stock prediction using SVR"""
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        """
        Initialize SVM model
        
        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            epsilon: Epsilon in epsilon-SVR model
            gamma: Kernel coefficient
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.best_params = None
        
    def build_model(self):
        """Build the SVM model"""
        print(f"Building SVM model with kernel: {self.kernel}")
        
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
            max_iter=config.SVM_CONFIG['max_iter'],
            cache_size=1000  # Increase cache for faster training
        )
        
        print("SVM model built successfully!")
        return self.model
    
    def train(self, X_train, y_train, verbose=1):
        """
        Train the SVM model
        
        Args:
            X_train: Training features
            y_train: Training target
            verbose: Verbosity level
            
        Returns:
            Trained model
        """
        if self.model is None:
            self.build_model()
        
        if verbose:
            print(f"\nTraining SVM model...")
            print(f"Training samples: {len(X_train)}")
            print(f"Kernel: {self.kernel}, C: {self.C}, epsilon: {self.epsilon}, gamma: {self.gamma}")
        
        self.model.fit(X_train, y_train)
        
        if verbose:
            print("Training completed!")
        
        return self.model
    
    def train_with_grid_search(self, X_train, y_train, X_val, y_val, param_grid=None, cv=3, verbose=1):
        """
        Train SVM with hyperparameter tuning using Grid Search
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (combined with train for CV)
            y_val: Validation target
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            verbose: Verbosity level
            
        Returns:
            Best model from grid search
        """
        if param_grid is None:
            param_grid = config.SVM_GRID_SEARCH
        
        print(f"\nPerforming Grid Search for SVM hyperparameters...")
        print(f"Parameter grid: {param_grid}")
        print(f"Cross-validation folds: {cv}")
        
        # Combine train and validation for cross-validation
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        
        # Create base model
        base_model = SVR(max_iter=config.SVM_CONFIG['max_iter'])
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            verbose=verbose,
            n_jobs=-1  # Use all available cores
        )
        
        grid_search.fit(X_combined, y_combined)
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"\nGrid Search completed!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score (neg MSE): {grid_search.best_score_:.6f}")
        
        # Update model parameters
        self.kernel = self.best_params['kernel']
        self.C = self.best_params['C']
        self.epsilon = self.best_params['epsilon']
        self.gamma = self.best_params['gamma']
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet!")
        
        return self.model.predict(X)
    
    def save_model(self, filepath=None):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if filepath is None:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            filepath = os.path.join(config.MODEL_DIR, f'svm_{self.kernel}_model.pkl')
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, f'svm_{self.kernel}_model.pkl')
        
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self):
        """Get information about the trained model"""
        if self.model is None:
            return "Model not trained yet"
        
        info = {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'n_support_vectors': self.model.n_support_[0] if hasattr(self.model, 'n_support_') else 'N/A'
        }
        
        if self.best_params:
            info['best_params'] = self.best_params
        
        return info


class SVMEnsemble:
    """Ensemble of SVM models with different kernels"""
    
    def __init__(self, kernels=['linear', 'rbf', 'poly']):
        """
        Initialize SVM ensemble
        
        Args:
            kernels: List of kernel types to use
        """
        self.kernels = kernels
        self.models = {}
        self.weights = None
        
    def build_models(self):
        """Build SVM models for each kernel"""
        print(f"Building SVM ensemble with kernels: {self.kernels}")
        
        for kernel in self.kernels:
            self.models[kernel] = SVMModel(
                kernel=kernel,
                C=config.SVM_CONFIG['C'],
                epsilon=config.SVM_CONFIG['epsilon'],
                gamma=config.SVM_CONFIG['gamma']
            )
            self.models[kernel].build_model()
        
        print("SVM ensemble built successfully!")
    
    def train(self, X_train, y_train, verbose=1):
        """
        Train all models in the ensemble
        
        Args:
            X_train: Training features
            y_train: Training target
            verbose: Verbosity level
        """
        if not self.models:
            self.build_models()
        
        print("\nTraining SVM ensemble...")
        
        for kernel, model in self.models.items():
            if verbose:
                print(f"\nTraining {kernel} kernel...")
            model.train(X_train, y_train, verbose=verbose)
        
        print("\nAll models in ensemble trained!")
    
    def predict(self, X, method='average'):
        """
        Make predictions using ensemble
        
        Args:
            X: Input features
            method: 'average' for simple average, 'weighted' for weighted average
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for kernel, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if method == 'average':
            return np.mean(predictions, axis=0)
        elif method == 'weighted' and self.weights is not None:
            return np.average(predictions, axis=0, weights=self.weights)
        else:
            return np.mean(predictions, axis=0)
    
    def set_weights(self, weights):
        """
        Set weights for weighted ensemble prediction
        
        Args:
            weights: List of weights for each model (must sum to 1)
        """
        if len(weights) != len(self.models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(self.models)})")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1, got {sum(weights)}")
        
        self.weights = weights
        print(f"Weights set: {dict(zip(self.kernels, weights))}")


if __name__ == '__main__':
    # Test the SVM models
    print("Testing SVM Models")
    print("=" * 50)
    
    # Test single SVM
    print("\n1. Testing Single SVM with RBF kernel")
    print("-" * 50)
    svm_rbf = SVMModel(kernel='rbf')
    svm_rbf.build_model()
    print(f"Model info: {svm_rbf.get_model_info()}")
    
    # Test SVM ensemble
    print("\n2. Testing SVM Ensemble")
    print("-" * 50)
    svm_ensemble = SVMEnsemble(kernels=['linear', 'rbf', 'poly'])
    svm_ensemble.build_models()
    
    print("\n" + "=" * 50)
    print("All SVM models built successfully!")
