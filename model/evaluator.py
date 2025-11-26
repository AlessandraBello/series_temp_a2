"""
Módulo para avaliação de modelos de séries temporais.
Cálculo de métricas como MAE, RMSE, MAPE, etc., e ferramentas
para avaliar a qualidade de transformações de séries.
"""
from typing import Dict, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .base import Evaluator


class TimeSeriesEvaluator(Evaluator):
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