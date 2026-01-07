# Quick Start Guide

## Getting Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Quick Test
```bash
python main.py --quick-test
```

This will:
- Load the first stock from the dataset
- Train a small ANN model (10 epochs)
- Display basic evaluation metrics

### 3. Train All Models on a Specific Stock
```bash
python main.py --ticker ACB
```

This will train and compare:
- Dense ANN
- LSTM
- SVM with RBF kernel
- SVM Ensemble

Results will be saved to `results/`, `plots/`, and `models/` directories.

## Alternative: Use the Jupyter Notebook

For interactive exploration and visualization:

```bash
jupyter notebook stock_forecasting_analysis.ipynb
```

The notebook provides:
- Step-by-step execution
- Inline visualizations
- Detailed explanations
- Easy experimentation

## Common Commands

### Train specific models only

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

### Advanced: SVM with Grid Search

Use hyperparameter optimization (slower but better results):
```bash
python main.py --ticker ACB --grid-search
```

## Project Structure

```
Project/
├── main.py                          # Main execution script
├── stock_forecasting_analysis.ipynb # Interactive notebook
├── config.py                        # Configuration settings
├── data_preprocessing.py            # Data preprocessing module
├── ann_models.py                    # ANN/LSTM models
├── svm_models.py                    # SVM models
├── evaluation.py                    # Evaluation metrics
├── dataset/                         # Data files
│   └── VN30_Dataset_2015_2026.csv
├── results/                         # Output results (generated)
├── models/                          # Saved models (generated)
└── plots/                           # Visualizations (generated)
```

## Customization

### Change Configuration

Edit `config.py` to customize:
- Training parameters (epochs, batch size)
- Model architectures
- Technical indicators
- Data split ratios

### Example: Change Number of Epochs

```python
# In config.py
ANN_CONFIG = {
    'epochs': 50,  # Change from 100 to 50
    'batch_size': 32,
    ...
}
```

## Available Tickers

The VN30 dataset includes various Vietnamese stock tickers. To see all available tickers:

```python
import pandas as pd
df = pd.read_csv('dataset/VN30_Dataset_2015_2026.csv')
print(df['Ticker'].unique())
```

## Troubleshooting

### Out of Memory
- Reduce batch size in `config.py`
- Use fewer features
- Select a shorter time period

### Training Too Slow
- Reduce number of epochs
- Use a smaller model architecture
- Try SVM instead (faster training)

### Poor Results
- Increase training data
- Add more technical indicators
- Use grid search for hyperparameter tuning
- Try ensemble methods

## Next Steps

1. ✅ Run quick test to verify installation
2. ✅ Train models on your chosen stock
3. ✅ Analyze results in the plots directory
4. ✅ Experiment with different configurations
5. ✅ Try the Jupyter notebook for deeper analysis
6. ✅ Read README.md for detailed documentation

## Need Help?

Check:
- `README.md` for comprehensive documentation
- Code comments in each module
- Jupyter notebook for step-by-step guide
- Paper reference for theoretical background

Happy forecasting! 📈
