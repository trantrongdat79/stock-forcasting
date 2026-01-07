"""
Artificial Neural Network (ANN) models for stock forecasting
Includes Dense ANN and LSTM architectures
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import config
import os


class ANNModel:
    """Base ANN model for stock prediction"""
    
    def __init__(self, input_shape, model_type='dense'):
        """
        Initialize ANN model
        
        Args:
            input_shape: Shape of input data
            model_type: 'dense' or 'lstm'
        """
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        self.history = None
        
        if model_type == 'dense':
            self.config = config.ANN_CONFIG
        elif model_type == 'lstm':
            self.config = config.LSTM_CONFIG
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def build_model(self):
        """Build the neural network architecture"""
        print(f"Building {self.model_type.upper()} model...")
        
        model = keras.Sequential()
        
        # Add input layer
        first_layer = True
        
        for layer_config in self.config['architecture']:
            layer_type = layer_config['type']
            
            if layer_type == 'dense':
                if first_layer:
                    model.add(layers.Dense(
                        layer_config['units'],
                        activation=layer_config['activation'],
                        input_shape=self.input_shape
                    ))
                    first_layer = False
                else:
                    model.add(layers.Dense(
                        layer_config['units'],
                        activation=layer_config['activation']
                    ))
            
            elif layer_type == 'lstm':
                if first_layer:
                    model.add(layers.LSTM(
                        layer_config['units'],
                        return_sequences=layer_config.get('return_sequences', False),
                        input_shape=self.input_shape
                    ))
                    first_layer = False
                else:
                    model.add(layers.LSTM(
                        layer_config['units'],
                        return_sequences=layer_config.get('return_sequences', False)
                    ))
            
            elif layer_type == 'dropout':
                model.add(layers.Dropout(layer_config['rate']))
        
        # Compile model
        model.compile(
            optimizer=self.config['optimizer'],
            loss=self.config['loss'],
            metrics=self.config['metrics']
        )
        
        self.model = model
        print(f"Model built successfully!")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, verbose=1):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Callbacks
        callbacks = []
        
        # Early stopping
        if 'early_stopping' in self.config:
            es_config = self.config['early_stopping']
            early_stop = EarlyStopping(
                monitor=es_config['monitor'],
                patience=es_config['patience'],
                restore_best_weights=es_config['restore_best_weights'],
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Model checkpoint
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        checkpoint_path = os.path.join(config.MODEL_DIR, f'{self.model_type}_best_model.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=verbose
        )
        
        print(f"\nTraining completed!")
        return self.history
    
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
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet!")
        
        print(f"\nEvaluating {self.model_type.upper()} model...")
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics_dict = {}
        for i, metric_name in enumerate(['loss'] + self.config['metrics']):
            metrics_dict[metric_name] = results[i]
            print(f"{metric_name}: {results[i]:.6f}")
        
        return metrics_dict
    
    def save_model(self, filepath=None):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if filepath is None:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            filepath = os.path.join(config.MODEL_DIR, f'{self.model_type}_model.h5')
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, f'{self.model_type}_model.h5')
        
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        
        return self.model.summary()


class DenseANN(ANNModel):
    """Dense (Feedforward) ANN for stock prediction"""
    
    def __init__(self, input_shape):
        super().__init__(input_shape, model_type='dense')


class LSTMModel(ANNModel):
    """LSTM model for stock prediction with sequences"""
    
    def __init__(self, input_shape):
        """
        Args:
            input_shape: Tuple (sequence_length, n_features)
        """
        super().__init__(input_shape, model_type='lstm')


if __name__ == '__main__':
    # Test the models
    print("Testing ANN Models")
    print("=" * 50)
    
    # Test Dense ANN
    print("\n1. Testing Dense ANN")
    print("-" * 50)
    input_shape_dense = (20,)  # 20 features
    dense_model = DenseANN(input_shape_dense)
    dense_model.build_model()
    dense_model.get_model_summary()
    
    # Test LSTM
    print("\n2. Testing LSTM")
    print("-" * 50)
    input_shape_lstm = (10, 20)  # 10 time steps, 20 features
    lstm_model = LSTMModel(input_shape_lstm)
    lstm_model.build_model()
    lstm_model.get_model_summary()
    
    print("\n" + "=" * 50)
    print("All model architectures built successfully!")
