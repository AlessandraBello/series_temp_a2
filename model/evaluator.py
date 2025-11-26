"""
evaluator.py

Módulo para avaliação de modelos de séries temporais.
Inclui métricas: MAE, RMSE, MAPE, MASE, RMSSE,
coerentes com as notas do curso.

"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .base import Evaluator


class TimeSeriesEvaluator(Evaluator):
    """
    Avaliador para previsões de séries temporais.

    Opção de usar erros escalados (MASE, RMSSE) com base em:
    - passeio aleatório (não sazonal)
    - passeio aleatório sazonal
    """

    def __init__(self, y_train: Optional[np.ndarray] = None, seasonal_periods: Optional[int] = None):
        super().__init__()
        self.metrics: List[str] = ["MAE", "RMSE", "MAPE", "MASE", "RMSSE"]
        self.y_train = None if y_train is None else np.asarray(y_train, dtype=float)
        self.m = seasonal_periods

    def _mase_denominator(self) -> Optional[float]:
        """
        Denominador do erro escalado:

        - Não sazonal: MAE do passeo aleatório (naive).
        - Sazonal: MAE do passeio aleatório sazonal.
        """
        if self.y_train is None:
            return None

        y = self.y_train

        if self.m is None or self.m == 1:
            if len(y) < 2:
                return None
            diffs = np.abs(np.diff(y))
            denom = np.mean(diffs)
        else:
            if len(y) <= self.m:
                return None
            diffs = np.abs(y[self.m:] - y[:-self.m])
            denom = np.mean(diffs)

        if denom == 0:
            return None
        return denom

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Avalia previsões com diversas métricas.

        Parameters
        ----------
        y_true : np.ndarray
            Valores reais.
        y_pred : np.ndarray
            Previsões.

        Returns
        -------
        Dict[str, float]
            Métricas calculadas.
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
            metrics["MAPE"] = float(np.mean(np.abs(e[mask] / y_true[mask])) * 100.0)
        else:
            metrics["MAPE"] = np.nan

        # MASE / RMSSE
        denom = self._mase_denominator()
        if denom is not None and denom > 0:
            q = e / denom
            metrics["MASE"] = float(np.mean(np.abs(q)))
            metrics["RMSSE"] = float(np.sqrt(np.mean(q**2)))
        else:
            metrics["MASE"] = np.nan
            metrics["RMSSE"] = np.nan

        return metrics
