"""
Evaluation and comparison module for stock forecasting models
Provides comprehensive metrics and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import config


class ModelEvaluator:
    """Evaluator for comparing different forecasting models"""
    
    def __init__(self):
        """Initialize the evaluator"""
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.PLOT_DIR, exist_ok=True)
        self.results = {}
    
    @staticmethod
    def calculate_mae(y_true, y_pred):
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def calculate_mse(y_true, y_pred):
        """Calculate Mean Squared Error"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """Calculate Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def calculate_r2(y_true, y_pred):
        """Calculate R-squared Score"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def calculate_directional_accuracy(y_true, y_pred):
        """
        Calculate Directional Accuracy (DA)
        Percentage of correct direction predictions
        """
        # Calculate changes in direction
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Calculate accuracy
        correct_directions = np.sum(true_direction == pred_direction)
        total_predictions = len(true_direction)
        
        return (correct_directions / total_predictions) * 100
    
    def evaluate_model(self, model_name, y_true, y_pred):
        """
        Evaluate a model and calculate all metrics
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'Model': model_name,
            'MAE': self.calculate_mae(y_true, y_pred),
            'MSE': self.calculate_mse(y_true, y_pred),
            'RMSE': self.calculate_rmse(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'R2': self.calculate_r2(y_true, y_pred),
            'DA': self.calculate_directional_accuracy(y_true, y_pred)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def print_evaluation(self, model_name):
        """Print evaluation results for a model"""
        if model_name not in self.results:
            print(f"No results for model: {model_name}")
            return
        
        metrics = self.results[model_name]
        print(f"\n{'='*60}")
        print(f"Evaluation Results for {model_name}")
        print(f"{'='*60}")
        print(f"MAE (Mean Absolute Error):           {metrics['MAE']:.6f}")
        print(f"MSE (Mean Squared Error):            {metrics['MSE']:.6f}")
        print(f"RMSE (Root Mean Squared Error):      {metrics['RMSE']:.6f}")
        print(f"MAPE (Mean Absolute % Error):        {metrics['MAPE']:.2f}%")
        print(f"R² (R-squared Score):                {metrics['R2']:.6f}")
        print(f"DA (Directional Accuracy):           {metrics['DA']:.2f}%")
        print(f"{'='*60}\n")
    
    def compare_models(self):
        """Compare all evaluated models"""
        if not self.results:
            print("No models to compare!")
            return None
        
        # Create comparison dataframe
        df = pd.DataFrame(self.results).T
        
        print(f"\n{'='*80}")
        print("Model Comparison Summary")
        print(f"{'='*80}")
        print(df.to_string())
        print(f"{'='*80}\n")
        
        # Find best model for each metric
        print("Best Models by Metric:")
        print("-" * 80)
        
        # For MAE, MSE, RMSE, MAPE: lower is better
        for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            best_model = df[metric].idxmin()
            best_value = df[metric].min()
            print(f"{metric:30s}: {best_model:20s} ({best_value:.6f})")
        
        # For R2, DA: higher is better
        for metric in ['R2', 'DA']:
            best_model = df[metric].idxmax()
            best_value = df[metric].max()
            print(f"{metric:30s}: {best_model:20s} ({best_value:.6f})")
        
        print(f"{'='*80}\n")
        
        # Save comparison to CSV
        output_path = os.path.join(config.OUTPUT_DIR, 'model_comparison.csv')
        df.to_csv(output_path)
        print(f"Comparison saved to {output_path}")
        
        return df
    
    def plot_predictions(self, y_true, predictions_dict, title='Model Predictions Comparison', 
                        save_path=None, n_samples=200):
        """
        Plot actual vs predicted values for multiple models
        
        Args:
            y_true: True values
            predictions_dict: Dictionary of {model_name: predictions}
            title: Plot title
            save_path: Path to save the plot
            n_samples: Number of samples to plot (for readability)
        """
        plt.figure(figsize=(15, 6))
        
        # Plot only last n_samples for clarity
        plot_range = slice(-n_samples, None)
        x = np.arange(len(y_true[plot_range]))
        
        # Plot true values
        plt.plot(x, y_true[plot_range], 'k-', label='Actual', linewidth=2, alpha=0.7)
        
        # Plot predictions for each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
        for (model_name, y_pred), color in zip(predictions_dict.items(), colors):
            plt.plot(x, y_pred[plot_range], '--', label=model_name, 
                    linewidth=1.5, alpha=0.7, color=color)
        
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Stock Price', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(config.PLOT_DIR, 'predictions_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")
        plt.close()
    
    def plot_error_distribution(self, y_true, predictions_dict, save_path=None):
        """
        Plot error distribution for multiple models
        
        Args:
            y_true: True values
            predictions_dict: Dictionary of {model_name: predictions}
            save_path: Path to save the plot
        """
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, y_pred) in zip(axes, predictions_dict.items()):
            errors = y_true - y_pred
            
            ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Prediction Error', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{model_name}\nMean Error: {np.mean(errors):.4f}', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(config.PLOT_DIR, 'error_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, save_path=None):
        """
        Plot bar charts comparing metrics across models
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            print("No results to plot!")
            return
        
        df = pd.DataFrame(self.results).T
        
        # Select metrics to plot
        metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'R2', 'DA']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            values = df[metric].values
            models = df.index.tolist()
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
            
            # Highlight best model
            if metric in ['MAE', 'RMSE', 'MAPE']:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
            
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(config.PLOT_DIR, 'metrics_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {save_path}")
        plt.close()
    
    def plot_scatter(self, y_true, predictions_dict, save_path=None):
        """
        Plot scatter plots of actual vs predicted for each model
        
        Args:
            y_true: True values
            predictions_dict: Dictionary of {model_name: predictions}
            save_path: Path to save the plot
        """
        n_models = len(predictions_dict)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for ax, (model_name, y_pred) in zip(axes, predictions_dict.items()):
            ax.scatter(y_true, y_pred, alpha=0.5, s=10)
            
            # Plot perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Values', fontsize=10)
            ax.set_ylabel('Predicted Values', fontsize=10)
            ax.set_title(f'{model_name}\nR² = {r2_score(y_true, y_pred):.4f}', 
                        fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(config.PLOT_DIR, 'scatter_comparison.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
        plt.close()
    
    def generate_report(self, y_true, predictions_dict, ticker=None):
        """
        Generate complete evaluation report with all plots and metrics
        
        Args:
            y_true: True values
            predictions_dict: Dictionary of {model_name: predictions}
            ticker: Stock ticker (optional)
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*80 + "\n")
        
        # Calculate metrics for all models
        for model_name, y_pred in predictions_dict.items():
            self.evaluate_model(model_name, y_true, y_pred)
            self.print_evaluation(model_name)
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        title_suffix = f" - {ticker}" if ticker else ""
        
        self.plot_predictions(y_true, predictions_dict, 
                            title=f'Stock Price Predictions{title_suffix}')
        self.plot_error_distribution(y_true, predictions_dict)
        self.plot_metrics_comparison()
        self.plot_scatter(y_true, predictions_dict)
        
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETE")
        print("="*80 + "\n")
        
        return comparison_df
    
    def generate_report_with_alignment(self, true_values_dict, predictions_dict, ticker=None):
        """
        Generate complete evaluation report with alignment for different-sized predictions
        Handles cases where different models produce different test set sizes (e.g., LSTM with sequences)
        
        Args:
            true_values_dict: Dictionary of {model_name: true_values}
            predictions_dict: Dictionary of {model_name: predictions}
            ticker: Stock ticker (optional)
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*80 + "\n")
        
        # Calculate metrics for each model with its corresponding true values
        for model_name in predictions_dict.keys():
            y_true = true_values_dict[model_name]
            y_pred = predictions_dict[model_name]
            print(f"\n{model_name}: {len(y_true)} test samples")
            self.evaluate_model(model_name, y_true, y_pred)
            self.print_evaluation(model_name)
        
        # Compare models
        comparison_df = self.compare_models()
        
        # For visualizations, align all predictions to the shortest length
        print("\nAligning predictions for visualization...")
        min_length = min(len(pred) for pred in predictions_dict.values())
        print(f"Using {min_length} common samples for visualization")
        
        # Align all arrays to the same length (use last min_length samples)
        aligned_predictions = {}
        aligned_true = None
        
        for model_name, y_pred in predictions_dict.items():
            y_true = true_values_dict[model_name]
            aligned_predictions[model_name] = y_pred[-min_length:]
            if aligned_true is None:
                aligned_true = y_true[-min_length:]
        
        # Generate plots with aligned data
        print("\nGenerating visualizations...")
        
        title_suffix = f" - {ticker}" if ticker else ""
        
        self.plot_predictions(aligned_true, aligned_predictions, 
                            title=f'Stock Price Predictions{title_suffix}')
        self.plot_error_distribution(aligned_true, aligned_predictions)
        self.plot_metrics_comparison()
        self.plot_scatter(aligned_true, aligned_predictions)
        
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETE")
        print("="*80 + "\n")
        
        return comparison_df


if __name__ == '__main__':
    # Test the evaluator
    print("Testing Model Evaluator")
    print("=" * 50)
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 100
    y_true = np.sin(np.linspace(0, 10, n_samples)) + np.random.normal(0, 0.1, n_samples)
    
    # Generate dummy predictions
    predictions = {
        'Model_A': y_true + np.random.normal(0, 0.15, n_samples),
        'Model_B': y_true + np.random.normal(0, 0.20, n_samples),
        'Model_C': y_true + np.random.normal(0, 0.10, n_samples)
    }
    
    # Create evaluator and generate report
    evaluator = ModelEvaluator()
    evaluator.generate_report(y_true, predictions, ticker='TEST')
    
    print("\nEvaluator test completed successfully!")
