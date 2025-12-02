"""
Classes base abstratas para o framework de séries temporais.
Define as interfaces que todos os componentes devem seguir.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd


class TimeSeriesModel(ABC):
    """
    Classe base abstrata para modelos de séries temporais.
    Define a interface comum que todos os modelos devem implementar.
    """
    
    def __init__(self, name: str = None):
        """
        Inicializa o modelo.
        
        Parameters
        ----------
        name : str, optional
            Nome do modelo para identificação
        """
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        self.fitted_values: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
    
    @abstractmethod
    def fit(self, y: np.ndarray, **kwargs) -> 'TimeSeriesModel':
        """
        Ajusta o modelo aos dados.
        
        Parameters
        ----------
        y : np.ndarray
            Série temporal univariada ou multivariada
        **kwargs
            Parâmetros adicionais específicos do modelo
            
        Returns
        -------
        TimeSeriesModel
            Retorna self para permitir method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Gera previsões para os próximos passos.
        
        Parameters
        ----------
        steps : int
            Número de passos à frente para prever
        **kwargs
            Parâmetros adicionais específicos do modelo
            
        Returns
        -------
        np.ndarray
            Array com as previsões
        """
        pass
    
    def predict_sample(self, steps: int, n_samples: int = 100, **kwargs) -> np.ndarray:
        """
        Gera amostras de previsões para os próximos passos.
        Importante para fazer a análise de incerteza e intervalos de confiança.
        
        Parameters
        ----------
        steps : int
            Número de passos à frente para prever
        n_samples : int
            Número de amostras a serem geradas
        **kwargs
            Parâmetros adicionais específicos do modelo
            
        Returns
        -------
        np.ndarray
            Array com as amostras de previsões, shape (n_samples, steps)
        """
        # Gerar amostras de previsões
        predictions = []
        for _ in range(n_samples):
            predictions.append(self.predict(steps, **kwargs))
        return np.array(predictions)
    
    def reset_model(self) -> None:
        """
        Reseta o estado do modelo, desfazendo o ajuste.
        """
        self.is_fitted = False
        self.fitted_values = None
        self.residuals = None
    
    def get_params(self) -> Dict[str, Any]:
        """
        Retorna os parâmetros do modelo.
        
        Returns
        -------
        Dict[str, Any]
            Dicionário com os parâmetros do modelo
        """
        return {}


class Preprocessor(ABC):
    """
    Classe base abstrata para preprocessamento de dados.
    Transformações como log, diferenciação, normalização etc.
    """
    
    @abstractmethod
    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Aplica a transformação.
        
        Parameters
        ----------
        y : np.ndarray
            Dados a ser transformados
            
        Returns
        -------
        np.ndarray
            Dados transformados
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Aplica a transformação inversa.
        
        Parameters
        ----------
        y : np.ndarray
            Dados transformados
            
        Returns
        -------
        np.ndarray
            Dados originais
        """
        pass


class Evaluator(ABC):
    """
    Classe base abstrata para avaliação de modelos.
    """
    
    def __init__(self):
        self.metrics: List[str] = []
    
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Avalia as previsões do modelo.
        
        Parameters
        ----------
        y_true : np.ndarray
            Valores reais
        y_pred : np.ndarray
            Valores previstos
            
        Returns
        -------
        Dict[str, float]
            Dicionário com as métricas calculadas. Chaves são os nomes das métricas.
        """
        pass


class Diagnostics(ABC):
    """
    Classe base abstrata para diagnósticos de modelos.
    """
    
    @abstractmethod
    def diagnose(self, model: TimeSeriesModel, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Realiza diagnósticos no modelo.
        
        Parameters
        ----------
        model : TimeSeriesModel
            Modelo a ser diagnosticado
        y : np.ndarray
            Dados observados
        **kwargs
            Parâmetros adicionais (caso necessário)
            
        Returns
        -------
        Dict[str, Any]
            Dicionário com os resultados dos diagnósticos. Chaves são os nomes das métricas.
        """
        pass


class Visualizer(ABC):
    """
    Classe base abstrata para visualização de séries temporais.
    """
    
    @abstractmethod
    def plot(self, y: np.ndarray, **kwargs) -> None:
        """
        Cria visualizações da série temporal através do matplotlib.
        
        Parameters
        ----------
        y : np.ndarray
            Série temporal a ser visualizada
        **kwargs
            Parâmetros adicionais para customização
        """
        pass

