"""
Módulo para visualização de séries temporais e resultados de modelos.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from .base import Visualizer, TimeSeriesModel


class TimeSeriesVisualizer(Visualizer):
    """
    Visualizador de séries temporais e resultados de modelos.
    """
    
    def __init__(self, figsize: tuple = (12, 6)):
        """
        Inicializa o visualizador.
        
        Parameters
        ----------
        figsize : tuple
            Tamanho das figuras (largura, altura)
        """
        self.figsize = figsize
    
    def plot(self, y: np.ndarray, **kwargs) -> None:
        """
        Plota uma série temporal.
        
        Parameters
        ----------
        y : np.ndarray
            Série temporal a ser visualizada
        **kwargs
            Parâmetros adicionais: title, xlabel, ylabel, label, etc.
        """
        plt.figure(figsize=self.figsize)
        plt.plot(y, label=kwargs.get('label', 'Series'))
        plt.title(kwargs.get('title', 'Time Series'))
        plt.xlabel(kwargs.get('xlabel', 'Time'))
        plt.ylabel(kwargs.get('ylabel', 'Value'))
        if kwargs.get('label'):
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_forecast(self, y_train: np.ndarray, y_forecast: np.ndarray,
                     y_true: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Plota a série original, previsões e valores reais (se disponíveis).
        
        Parameters
        ----------
        y_train : np.ndarray
            Série temporal de treino
        y_forecast : np.ndarray
            Previsões do modelo
        y_true : np.ndarray, optional
            Valores reais para comparação
        **kwargs
            Parâmetros adicionais para customização
        """
        plt.figure(figsize=self.figsize)
        
        # Série de treino
        train_indices = np.arange(len(y_train))
        plt.plot(train_indices, y_train, label='Training', color='blue', alpha=0.7)
        
        # Previsões
        forecast_start = len(y_train)
        forecast_indices = np.arange(forecast_start, forecast_start + len(y_forecast))
        plt.plot(forecast_indices, y_forecast, label='Forecast', color='red', 
                linestyle='--', marker='o', markersize=4)
        
        # Valores reais (se disponíveis)
        if y_true is not None:
            true_indices = np.arange(forecast_start, forecast_start + len(y_true))
            plt.plot(true_indices, y_true, label='Actual', color='green', 
                    marker='s', markersize=4)
        
        plt.axvline(x=forecast_start, color='gray', linestyle=':', alpha=0.5, 
                   label='Forecast Start')
        plt.title(kwargs.get('title', 'Time Series Forecast'))
        plt.xlabel(kwargs.get('xlabel', 'Time'))
        plt.ylabel(kwargs.get('ylabel', 'Value'))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, residuals: np.ndarray, **kwargs) -> None:
        """
        Plota os resíduos do modelo.
        
        Parameters
        ----------
        residuals : np.ndarray
            Resíduos do modelo
        **kwargs
            Parâmetros adicionais para customização
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Resíduos ao longo do tempo
        axes[0, 0].plot(residuals)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histograma dos resíduos
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot (aproximado)
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.linspace(-3, 3, len(sorted_residuals))
        theoretical_values = theoretical_quantiles * np.std(residuals) + np.mean(residuals)
        axes[1, 0].scatter(theoretical_values, sorted_residuals, alpha=0.5)
        axes[1, 0].plot([theoretical_values.min(), theoretical_values.max()],
                       [theoretical_values.min(), theoretical_values.max()],
                       'r--', alpha=0.5)
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].set_xlabel('Theoretical Quantiles')
        axes[1, 0].set_ylabel('Sample Quantiles')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Autocorrelação dos resíduos
        max_lag = min(20, len(residuals) // 2)
        autocorrs = [self._autocorr(residuals, lag) for lag in range(1, max_lag + 1)]
        axes[1, 1].bar(range(1, max_lag + 1), autocorrs)
        axes[1, 1].axhline(y=0, color='r', linestyle='-', alpha=0.5)
        axes[1, 1].set_title('Residuals Autocorrelation')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _autocorr(self, x: np.ndarray, lag: int) -> float:
        """Calcula a autocorrelação em um lag específico"""
        if lag >= len(x):
            return 0.0
        x_shifted = x[lag:]
        x_original = x[:-lag]
        if np.std(x_original) == 0 or np.std(x_shifted) == 0:
            return 0.0
        return np.corrcoef(x_original, x_shifted)[0, 1]