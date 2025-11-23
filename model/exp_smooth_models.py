"""
Implementação de modelos de Suavização Exponencial.
"""

from .base import TimeSeriesModel
import numpy as np
from typing import Dict, Any, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import warnings


class ETSModel(TimeSeriesModel):
    """
    Classe base para modelos ETS (Error, Trend, Seasonality).
    Adiciona funcionalidades específicas do statsmodels à classe abstrata.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._model = None
        self._fitted_model = None
    
    def predict_sample(self, steps: int, n_samples: int = 100, **kwargs) -> np.ndarray:
        """
        Sobrescreve o método base para usar simulação do statsmodels.
        Gera amostras de previsões usando bootstrap dos resíduos.
        
        Parameters
        ----------
        steps : int
            Número de passos à frente para prever
        n_samples : int
            Número de amostras a serem geradas
        **kwargs
            Parâmetros adicionais
            
        Returns
        -------
        np.ndarray
            Array com as amostras de previsões, shape (n_samples, steps)
        """
        if not self.is_fitted:
            raise ValueError("O modelo precisa ser ajustado antes de gerar previsões.")
        
        # Usa o método simulate do statsmodels para gerar amostras
        predictions = []
        for _ in range(n_samples):
            # Simula trajetórias futuras com ruído
            sim = self._fitted_model.simulate(
                nsimulations=steps,
                anchor='end',
                repetitions=1,
                random_errors='bootstrap'
            )
            predictions.append(sim)
        
        return np.array(predictions)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Sobrescreve para retornar os parâmetros estimados do statsmodels.
        
        Returns
        -------
        Dict[str, Any]
            Dicionário com os parâmetros estimados do modelo
        """
        if not self.is_fitted:
            return {}
        return self._fitted_model.params
    
    def _post_fit(self, y):
        """
        Executa operações padronizadas após o ajuste de um modelo ETS.

        Este método armazena os valores ajustados pelo modelo, calcula 
        os resíduos e marca o objeto como ajustado.

        Parameters
        ----------
        y : np.ndarray
            Série temporal utilizada no ajuste do modelo.
        """
        self.fitted_values = self._fitted_model.fittedvalues
        self.residuals = y - self.fitted_values
        self.is_fitted = True


class SimpleExponentialSmoothingModel(ETSModel):
    """
    Suavização Exponencial Simples - ETS(A,N,N).
    
    Adequado para séries sem tendência ou sazonalidade.
    Atribui pesos decrescentes exponencialmente às observações passadas.
    
    Equações:
    - Previsão: y_hat(t+h|t) = l_t
    - Nível: l_t = alpha*y_t + (1-alpha)*l_(t-1)
    
    Parâmetros:
    - alpha (smoothing_level): controla o peso das observações recentes (0 < alpha <= 1)
      * alpha próximo de 1: mais peso às observações recentes (similar ao método ingênuo)
      * alpha próximo de 0: pesos quase iguais (similar à média simples)
    """
    
    def __init__(self, smoothing_level: Optional[float] = None, name: str = None):
        """
        Parameters
        ----------
        smoothing_level : float, optional
            Parâmetro alpha de suavização. Se None, será estimado automaticamente.
        name : str, optional
            Nome do modelo
        """
        super().__init__(name or "Simple Exponential Smoothing")
        self.smoothing_level = smoothing_level
    
    def fit(self, y: np.ndarray, **kwargs) -> 'SimpleExponentialSmoothingModel':
        """
        Ajusta o modelo de Suavização Exponencial Simples.
        
        Parameters
        ----------
        y : np.ndarray
            Série temporal univariada
        **kwargs
            Parâmetros adicionais para o método fit do statsmodels
            
        Returns
        -------
        SimpleExponentialSmoothingModel
            Retorna self para method chaining
        """
        self._model = SimpleExpSmoothing(y)
        
        # Parâmetros para o fit
        fit_params = kwargs.copy()
        if self.smoothing_level is not None:
            fit_params['smoothing_level'] = self.smoothing_level
        
        self._fitted_model = self._model.fit(**fit_params)
        
        # Operações após o ajuste do modelo
        self._post_fit(y)
        
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Gera previsões para os próximos passos.
        Como não há tendência, a previsão é constante = l_T
        
        Parameters
        ----------
        steps : int
            Número de passos à frente
        **kwargs
            Parâmetros adicionais (não utilizados)
            
        Returns
        -------
        np.ndarray
            Array com as previsões
        """
        if not self.is_fitted:
            raise ValueError("O modelo precisa ser ajustado antes de fazer previsões.")
        
        return self._fitted_model.forecast(steps)


class HoltLinearTrendModel(ETSModel):
    """
    Método de Holt para Tendência Linear - ETS(A,A,N) ou ETS(A,Ad,N).
    
    Extensão da suavização exponencial que considera nível e tendência.
    
    Equações:
    - Previsão: y_hat(T+h|T) = l_T + h*b_T (linear)
    - Previsão amortecida: y_hat(T+h|T) = l_T + (phi + phi^2 + ... + phi^h)*b_T
    - Nível: l_t = alpha*y_t + (1-alpha)*(l_(t-1) + b_(t-1))
    - Tendência: b_t = beta*(l_t - l_(t-1)) + (1-beta)*b_(t-1)
    
    Parâmetros:
    - alpha (smoothing_level): controla o nível (0 < alpha <= 1)
    - beta (smoothing_trend): controla a tendência (0 <= beta <= 1)
    - phi (damping_trend): amortecimento da tendência (0 < phi < 1), tipicamente 0.8-0.98
    """
    
    def __init__(
        self,
        smoothing_level: Optional[float] = None,
        smoothing_trend: Optional[float] = None,
        damped_trend: bool = False,
        damping_trend: Optional[float] = None,
        name: str = None
    ):
        """
        Parameters
        ----------
        smoothing_level : float, optional
            Parâmetro alpha de suavização do nível
        smoothing_trend : float, optional
            Parâmetro beta de suavização da tendência
        damped_trend : bool, default=False
            Se True, usa tendência amortecida (ETS(A,Ad,N))
        damping_trend : float, optional
            Parâmetro phi de amortecimento (0 < phi < 1)
            Na prática, geralmente entre 0.8 e 0.98
        name : str, optional
            Nome do modelo
        """
        model_name = name or ("Holt Damped Trend" if damped_trend else "Holt Linear Trend")
        super().__init__(model_name)
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.damped_trend = damped_trend
        self.damping_trend = damping_trend
    
    def fit(self, y: np.ndarray, **kwargs) -> 'HoltLinearTrendModel':
        """
        Ajusta o modelo de Holt.
        
        Parameters
        ----------
        y : np.ndarray
            Série temporal univariada
        **kwargs
            Parâmetros adicionais para o método fit
            
        Returns
        -------
        HoltLinearTrendModel
            Retorna self para method chaining
        """
        self._model = Holt(y, damped_trend=self.damped_trend)
        
        # Parâmetros para o fit
        fit_params = kwargs.copy()
        if self.smoothing_level is not None:
            fit_params['smoothing_level'] = self.smoothing_level
        if self.smoothing_trend is not None:
            fit_params['smoothing_trend'] = self.smoothing_trend
        if self.damping_trend is not None and self.damped_trend:
            fit_params['damping_trend'] = self.damping_trend
        
        self._fitted_model = self._model.fit(**fit_params)
        
        # Operações após o ajuste do modelo
        self._post_fit(y)
        
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Gera previsões para os próximos passos.
        
        Parameters
        ----------
        steps : int
            Número de passos à frente
        **kwargs
            Parâmetros adicionais (não utilizados)
            
        Returns
        -------
        np.ndarray
            Array com as previsões
        """
        if not self.is_fitted:
            raise ValueError("O modelo precisa ser ajustado antes de fazer previsões.")
        
        return self._fitted_model.forecast(steps)


class HoltWintersAdditiveModel(ETSModel):
    """
    Método de Holt-Winters Aditivo - ETS(A,A,A).
    
    Para séries com tendência e sazonalidade aditiva (variações sazonais constantes).
    
    Equações:
    - Previsão: y_hat(T+h|T) = l_T + h*b_T + S_(T+h-m(k+1))
    - Nível: l_t = alpha*(y_t - S_(t-m)) + (1-alpha)*(l_(t-1) + b_(t-1))
    - Tendência: b_t = beta*(l_t - l_(t-1)) + (1-beta)*b_(t-1)
    - Sazonalidade: S_t = gamma*(y_t - l_(t-1) - b_(t-1)) + (1-gamma)*S_(t-m)
    
    Parâmetros:
    - alpha (smoothing_level): nível (0 < alpha <= 1)
    - beta (smoothing_trend): tendência (0 < beta <= 1)
    - gamma (smoothing_seasonal): sazonalidade (0 < gamma <= 1-alpha)
    
    Restrição: 0 < gamma <= 1-alpha garante balanceamento entre ajustes de nível e sazonalidade.
    """
    
    def __init__(
        self,
        seasonal_periods: int,
        smoothing_level: Optional[float] = None,
        smoothing_trend: Optional[float] = None,
        smoothing_seasonal: Optional[float] = None,
        damped_trend: bool = False,
        name: str = None
    ):
        """
        Parameters
        ----------
        seasonal_periods : int
            Número de períodos sazonais (m)
            Exemplos: 12 para dados mensais com sazonalidade anual,
                     4 para dados trimestrais, 7 para dados diários com ciclo semanal
        smoothing_level : float, optional
            Parâmetro alpha de suavização do nível
        smoothing_trend : float, optional
            Parâmetro beta de suavização da tendência
        smoothing_seasonal : float, optional
            Parâmetro gamma de suavização da sazonalidade
        damped_trend : bool, default=False
            Se True, usa tendência amortecida
        name : str, optional
            Nome do modelo
        """
        super().__init__(name or "Holt-Winters Additive")
        self.seasonal_periods = seasonal_periods
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.damped_trend = damped_trend
    
    def fit(self, y: np.ndarray, **kwargs) -> 'HoltWintersAdditiveModel':
        """
        Ajusta o modelo de Holt-Winters Aditivo.
        
        Parameters
        ----------
        y : np.ndarray
            Série temporal univariada
            Deve ter pelo menos 2 * seasonal_periods observações
        **kwargs
            Parâmetros adicionais para o método fit
            
        Returns
        -------
        HoltWintersAdditiveModel
            Retorna self para method chaining
        """
        self._model = ExponentialSmoothing(
            y,
            trend='add',
            seasonal='add',
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend
        )
        
        # Parâmetros para o fit
        fit_params = kwargs.copy()
        if self.smoothing_level is not None:
            fit_params['smoothing_level'] = self.smoothing_level
        if self.smoothing_trend is not None:
            fit_params['smoothing_trend'] = self.smoothing_trend
        if self.smoothing_seasonal is not None:
            fit_params['smoothing_seasonal'] = self.smoothing_seasonal
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = self._model.fit(**fit_params)
        
        # Operações após o ajuste do modelo
        self._post_fit(y)
        
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Gera previsões para os próximos passos.
        
        Parameters
        ----------
        steps : int
            Número de passos à frente
        **kwargs
            Parâmetros adicionais (não utilizados)
            
        Returns
        -------
        np.ndarray
            Array com as previsões
        """
        if not self.is_fitted:
            raise ValueError("O modelo precisa ser ajustado antes de fazer previsões.")
        
        return self._fitted_model.forecast(steps)


class HoltWintersMultiplicativeModel(ETSModel):
    """
    Método de Holt-Winters Multiplicativo - ETS(A,A,M).
    
    Para séries onde as variações sazonais mudam proporcionalmente ao nível.
    Preferível quando a amplitude da sazonalidade aumenta/diminui com o nível da série.
    
    Equações:
    - Previsão: y_hat(T+h|T) = (l_T + h*b_T) * S_(T+h-m(k+1))
    - Nível: l_t = alpha*(y_t/S_(t-m)) + (1-alpha)*(l_(t-1) + b_(t-1))
    - Tendência: b_t = beta*(l_t - l_(t-1)) + (1-beta)*b_(t-1)
    - Sazonalidade: S_t = gamma*(y_t/(l_(t-1) + b_(t-1))) + (1-gamma)*S_(t-m)
    
    Parâmetros:
    - alpha (smoothing_level): nível (0 < alpha <= 1)
    - beta (smoothing_trend): tendência (0 < beta <= 1)
    - gamma (smoothing_seasonal): sazonalidade (0 < gamma < 1)
    """
    
    def __init__(
        self,
        seasonal_periods: int,
        smoothing_level: Optional[float] = None,
        smoothing_trend: Optional[float] = None,
        smoothing_seasonal: Optional[float] = None,
        damped_trend: bool = False,
        name: str = None
    ):
        """
        Parameters
        ----------
        seasonal_periods : int
            Número de períodos sazonais (m)
            Exemplos: 12 para dados mensais com sazonalidade anual,
                     4 para dados trimestrais, 7 para dados diários com ciclo semanal
        smoothing_level : float, optional
            Parâmetro alpha de suavização do nível
        smoothing_trend : float, optional
            Parâmetro beta de suavização da tendência
        smoothing_seasonal : float, optional
            Parâmetro gamma de suavização da sazonalidade
        damped_trend : bool, default=False
            Se True, usa tendência amortecida
        name : str, optional
            Nome do modelo
        """
        super().__init__(name or "Holt-Winters Multiplicative")
        self.seasonal_periods = seasonal_periods
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.damped_trend = damped_trend
    
    def fit(self, y: np.ndarray, **kwargs) -> 'HoltWintersMultiplicativeModel':
        """
        Ajusta o modelo de Holt-Winters Multiplicativo.
        
        Parameters
        ----------
        y : np.ndarray
            Série temporal univariada
            Deve ter pelo menos 2 * seasonal_periods observações
            IMPORTANTE: Não pode conter valores zero ou negativos
        **kwargs
            Parâmetros adicionais para o método fit
            
        Returns
        -------
        HoltWintersMultiplicativeModel
            Retorna self para method chaining
        """
        self._model = ExponentialSmoothing(
            y,
            trend='add',
            seasonal='mul',
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend
        )
        
        # Parâmetros para o fit
        fit_params = kwargs.copy()
        if self.smoothing_level is not None:
            fit_params['smoothing_level'] = self.smoothing_level
        if self.smoothing_trend is not None:
            fit_params['smoothing_trend'] = self.smoothing_trend
        if self.smoothing_seasonal is not None:
            fit_params['smoothing_seasonal'] = self.smoothing_seasonal
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_model = self._model.fit(**fit_params)
        
        # Operações após o ajuste do modelo
        self._post_fit(y)
        
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Gera previsões para os próximos passos.
        
        Parameters
        ----------
        steps : int
            Número de passos à frente
        **kwargs
            Parâmetros adicionais (não utilizados)
            
        Returns
        -------
        np.ndarray
            Array com as previsões
        """
        if not self.is_fitted:
            raise ValueError("O modelo precisa ser ajustado antes de fazer previsões.")
        
        return self._fitted_model.forecast(steps)
    