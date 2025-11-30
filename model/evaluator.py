"""
Módulo para avaliação de modelos de séries temporais.
Cálculo de métricas como MAE, RMSE, MAPE, etc., e ferramentas
para avaliar a qualidade de transformações de séries.
"""
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .base import Evaluator


class TransformEvaluator(Evaluator):
    """
    Implementação concreta de avaliador de séries temporais.

    - Para transformações (como diferença, log-diferença), expõe
      o método `evaluate`, que usa o teste ADF para
      medir estacionariedade.
    """

    def __init__(self) -> None:
        super().__init__()
    
    def evaluate(
        self, series: pd.Series, transform_name: str, series_name: str
    ) -> Dict[str, Any]:
        """
        Avalia a qualidade de uma transformação de série temporal
        através do teste ADF (Augmented Dickey-Fuller).

        Parameters
        ----------
        series : pd.Series
            Série já transformada (por exemplo, diferença, log-diferença).
        transform_name : str
            Nome da transformação aplicada.
        series_name : str
            Nome da série original.

        Returns
        -------
        Dict[str, Any]
            Dicionário com p-valor, estatística do teste e número de
            observações utilizadas.
        """
        clean_series = series.dropna()
        result = adfuller(clean_series.values)

        return {
            "series": series_name,
            "transform": transform_name,
            "p_value": float(result[1]),
            "test_stat": float(result[0]),
            "n_obs": int(result[3]),
        }
            
class TimeSeriesEvaluator(Evaluator):
    """
    Implementação concreta de avaliador de séries temporais.

    Avalia previsões de séries temporais utilizando métricas como MAE, RMSE, MAPE, MASE, etc.
    """

    def __init__(self) -> None:
        super().__init__()

    # --------- AVALIAÇÃO DE PREVISÃO (MÉTRICAS) ---------
    def _mase_denominator(
        self,
        y_train: Optional[np.ndarray],
        seasonal_periods: Optional[int] = None
    ) -> Optional[float]:
        """
        Denominador para MASE/RMSSE, baseado em:

        - passeio aleatório (não sazonal) ou
        - passeio aleatório sazonal (quando seasonal_periods é fornecido).
        """
        if y_train is None:
            return None

        y = np.asarray(y_train, dtype=float)

        if seasonal_periods is None or seasonal_periods == 1:
            if len(y) < 2:
                return None
            diffs = np.abs(np.diff(y))
        else:
            m = seasonal_periods
            if len(y) <= m:
                return None
            diffs = np.abs(y[m:] - y[:-m])

        denom = float(np.mean(diffs))
        if denom == 0.0:
            return None
        return denom

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        seasonal_periods: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Avalia previsões de um modelo (fora da amostra).

        Parameters
        ----------
        y_true : np.ndarray
            Valores reais.
        y_pred : np.ndarray
            Previsões do modelo.
        y_train : np.ndarray, optional
            Série de treino, usada para computar MASE/RMSSE.
        seasonal_periods : int, optional
            Período sazonal m, se aplicável (para MASE/RMSSE sazonais).

        Returns
        -------
        Dict[str, float]
            Métricas: MAE, RMSE, MAPE, MASE, RMSSE.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true e y_pred devem ter o mesmo shape.")

        e = y_true - y_pred
        metrics: Dict[str, float] = {}

        # MAE
        metrics["MAE"] = float(np.mean(np.abs(e)))

        # RMSE
        metrics["RMSE"] = float(np.sqrt(np.mean(e**2)))

        # MAPE (com proteção para y_true == 0)
        mask = y_true != 0
        if np.any(mask):
            metrics["MAPE"] = float(
                np.mean(np.abs(e[mask] / y_true[mask])) * 100.0
            )
        else:
            metrics["MAPE"] = np.nan

        # MASE / RMSSE
        denom = self._mase_denominator(y_train=y_train, seasonal_periods=seasonal_periods)
        if denom is not None and denom > 0:
            scaled_error = e / denom
            metrics["MASE"] = float(np.mean(np.abs(scaled_error)))
            metrics["RMSSE"] = float(np.sqrt(np.mean(scaled_error**2)))
        else:
            metrics["MASE"] = np.nan
            metrics["RMSSE"] = np.nan

        return metrics
