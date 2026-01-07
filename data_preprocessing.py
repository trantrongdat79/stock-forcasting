"""
Data preprocessing module for stock forecasting
Handles data loading, cleaning, feature engineering, and preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ta
import config


class StockDataPreprocessor:
    """Preprocessor for stock market data"""
    
    def __init__(self, data_path=None):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the CSV file containing stock data
        """
        self.data_path = data_path or config.DATA_PATH
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.data = None
        self.feature_columns = None
        
    def load_data(self, ticker=None):
        """
        Load stock data from CSV file
        
        Args:
            ticker: Specific stock ticker to filter (optional)
            
        Returns:
            DataFrame with stock data
        """
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data = self.data.sort_values('time')
        
        if ticker:
            self.data = self.data[self.data['Ticker'] == ticker].copy()
            print(f"Filtered data for ticker: {ticker}")
        
        print(f"Loaded {len(self.data)} records")
        return self.data
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators as features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        print("Adding technical indicators...")
        df = df.copy()
        
        # Simple Moving Averages
        for window in config.TECHNICAL_INDICATORS['SMA']:
            df[f'SMA_{window}'] = ta.trend.sma_indicator(df['close'], window=window)
        
        # Exponential Moving Averages
        for window in config.TECHNICAL_INDICATORS['EMA']:
            df[f'EMA_{window}'] = ta.trend.ema_indicator(df['close'], window=window)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['close'], window=config.TECHNICAL_INDICATORS['RSI'])
        
        # MACD
        if config.TECHNICAL_INDICATORS['MACD']:
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=config.TECHNICAL_INDICATORS['BB'])
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_width'] = bb.bollinger_wband()
        
        # On-Balance Volume
        if config.TECHNICAL_INDICATORS['OBV']:
            df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=10).std()
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Drop NaN values created by indicators
        df = df.dropna()
        
        print(f"Added {df.shape[1] - 7} technical indicators")
        return df
    
    def prepare_features(self, df, target_col='close'):
        """
        Prepare features and target for modeling
        
        Args:
            df: DataFrame with all features
            target_col: Column to predict
            
        Returns:
            X (features), y (target)
        """
        # Define feature columns (exclude time, ticker, and target)
        exclude_cols = ['time', 'Ticker', target_col]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].values
        y = df[target_col].values
        
        return X, y
    
    def create_sequences(self, X, y, sequence_length=None):
        """
        Create sequences for time series prediction
        
        Args:
            X: Feature array
            y: Target array
            sequence_length: Number of time steps to look back
            
        Returns:
            X_seq, y_seq: Sequenced data
        """
        if sequence_length is None:
            sequence_length = config.SEQUENCE_LENGTH
            
        X_seq, y_seq = [], []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def split_data(self, X, y, test_size=None, validation_split=None):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of data for testing
            validation_split: Proportion of training data for validation
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        if test_size is None:
            test_size = 1 - config.TRAIN_TEST_SPLIT
        if validation_split is None:
            validation_split = config.VALIDATION_SPLIT
            
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_split, shuffle=False
        )
        
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Scale features and target using MinMaxScaler
        
        Args:
            Train, validation, and test sets
            
        Returns:
            Scaled train, validation, and test sets
        """
        # Reshape for scaling if needed
        train_shape = X_train.shape
        val_shape = X_val.shape
        test_shape = X_test.shape
        
        if len(train_shape) == 3:  # Sequence data
            X_train_2d = X_train.reshape(-1, train_shape[-1])
            X_val_2d = X_val.reshape(-1, val_shape[-1])
            X_test_2d = X_test.reshape(-1, test_shape[-1])
        else:
            X_train_2d = X_train
            X_val_2d = X_val
            X_test_2d = X_test
        
        # Fit scaler on training data only
        self.scaler_features.fit(X_train_2d)
        self.scaler_target.fit(y_train.reshape(-1, 1))
        
        # Transform all sets
        X_train_scaled = self.scaler_features.transform(X_train_2d)
        X_val_scaled = self.scaler_features.transform(X_val_2d)
        X_test_scaled = self.scaler_features.transform(X_test_2d)
        
        # Reshape back if needed
        if len(train_shape) == 3:
            X_train_scaled = X_train_scaled.reshape(train_shape)
            X_val_scaled = X_val_scaled.reshape(val_shape)
            X_test_scaled = X_test_scaled.reshape(test_shape)
        
        y_train_scaled = self.scaler_target.transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_target.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_target.transform(y_test.reshape(-1, 1)).flatten()
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
    
    def inverse_transform_target(self, y):
        """
        Inverse transform scaled target values
        
        Args:
            y: Scaled target values
            
        Returns:
            Original scale values
        """
        return self.scaler_target.inverse_transform(y.reshape(-1, 1)).flatten()
    
    def preprocess_for_svm(self, ticker=None):
        """
        Complete preprocessing pipeline for SVM (no sequences)
        
        Args:
            ticker: Stock ticker to process
            
        Returns:
            Prepared and scaled data for SVM
        """
        # Load and prepare data
        df = self.load_data(ticker)
        df = self.add_technical_indicators(df)
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Scale data
        X_train, X_val, X_test, y_train, y_val, y_test = self.scale_data(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_columns': self.feature_columns
        }
    
    def preprocess_for_ann(self, ticker=None, use_sequences=False):
        """
        Complete preprocessing pipeline for ANN
        
        Args:
            ticker: Stock ticker to process
            use_sequences: Whether to create sequences for LSTM/RNN
            
        Returns:
            Prepared and scaled data for ANN
        """
        # Load and prepare data
        df = self.load_data(ticker)
        df = self.add_technical_indicators(df)
        X, y = self.prepare_features(df)
        
        # Create sequences if requested
        if use_sequences:
            X, y = self.create_sequences(X, y)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Scale data
        X_train, X_val, X_test, y_train, y_val, y_test = self.scale_data(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_columns': self.feature_columns
        }


if __name__ == '__main__':
    # Test the preprocessor
    preprocessor = StockDataPreprocessor()
    
    # Get unique tickers
    df = pd.read_csv(config.DATA_PATH)
    tickers = df['Ticker'].unique()
    print(f"Available tickers: {tickers[:10]}...")  # Show first 10
    
    # Test with first ticker
    if len(tickers) > 0:
        test_ticker = tickers[0]
        print(f"\nTesting with ticker: {test_ticker}")
        
        # Test SVM preprocessing
        print("\n=== SVM Preprocessing ===")
        svm_data = preprocessor.preprocess_for_svm(ticker=test_ticker)
        print(f"SVM Training shape: {svm_data['X_train'].shape}")
        print(f"Features: {len(svm_data['feature_columns'])}")
        
        # Test ANN preprocessing (no sequences)
        print("\n=== ANN Preprocessing (Dense) ===")
        preprocessor2 = StockDataPreprocessor()
        ann_data = preprocessor2.preprocess_for_ann(ticker=test_ticker, use_sequences=False)
        print(f"ANN Training shape: {ann_data['X_train'].shape}")
        
        # Test LSTM preprocessing (with sequences)
        print("\n=== LSTM Preprocessing (Sequences) ===")
        preprocessor3 = StockDataPreprocessor()
        lstm_data = preprocessor3.preprocess_for_ann(ticker=test_ticker, use_sequences=True)
        print(f"LSTM Training shape: {lstm_data['X_train'].shape}")
