"""
Módulo com implementações de modelos de séries temporais.
Este módulo contém exemplos de implementação que seguem a interface TimeSeriesModel.
"""
import numpy as np
from typing import Optional
from .base import TimeSeriesModel


class NaiveForecast(TimeSeriesModel):
    """
    Modelo ingênuo: prevê o último valor observado.
    Útil como baseline.
    """
    
    def __init__(self):
        super().__init__("NaiveForecast")
        self.last_value: Optional[float] = None
    
    def fit(self, y: np.ndarray, **kwargs) -> 'NaiveForecast':
        """
        Ajusta o modelo (simplesmente armazena o último valor).
        """
        if len(y) == 0:
            raise ValueError("y must not be empty")
        self.last_value = y[-1]
        self.is_fitted = True
        self.fitted_values = np.full_like(y, self.last_value)
        self.residuals = y - self.fitted_values
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Previsão: repete o último valor observado.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return np.full(steps, self.last_value)


class MovingAverage(TimeSeriesModel):
    """
    Modelo de média móvel simples.
    """
    
    def __init__(self, window: int = 3):
        """
        Inicializa o modelo de média móvel.
        
        Parameters
        ----------
        window : int
            Tamanho da janela da média móvel
        """
        super().__init__("MovingAverage")
        self.window = window
        self.last_values: Optional[np.ndarray] = None
    
    def fit(self, y: np.ndarray, **kwargs) -> 'MovingAverage':
        """
        Ajusta o modelo calculando a média móvel.
        """
        if len(y) < self.window:
            raise ValueError(f"y must have at least {self.window} values")
        
        # Calcula média móvel através de convolução
        self.fitted_values = np.convolve(y, np.ones(self.window)/self.window, mode='valid')
        # Preenche os primeiros valores com o primeiro valor calculado
        padding = np.full(self.window - 1, self.fitted_values[0])
        self.fitted_values = np.concatenate([padding, self.fitted_values])
        
        self.last_values = y[-self.window:]
        self.residuals = y - self.fitted_values
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Previsão: usa a média dos últimos valores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = []
        current_window = self.last_values.copy()
        
        for _ in range(steps):
            next_value = np.mean(current_window)
            forecast.append(next_value)
            # Atualiza a janela (remove o primeiro, adiciona a previsão)
            current_window = np.concatenate([current_window[1:], [next_value]])
        
        return np.array(forecast)
    
    def get_params(self) -> dict:
        """Retorna os parâmetros do modelo"""
        return {'window': self.window}