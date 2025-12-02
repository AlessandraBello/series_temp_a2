"""
diagnostics.py

Diagnósticos de modelos de séries temporais:
- Resumo de resíduos
- Teste de Ljung-Box para autocorrelação dos resíduos
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from .base import Diagnostics, TimeSeriesModel
from typing import Dict, Any


class ResidualDiagnostics(Diagnostics):
    """
    Realiza diagnósticos básicos em resíduos:
    - média e variância
    - teste de Ljung-Box
    - autocorrelação até um certo lag
    """

    def __init__(self, lags: int = 24):
        self.lags = lags

    def diagnose(self, model: TimeSeriesModel, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        if model.residuals is None:
            raise ValueError("Model has no residuals. Fit the model first.")

        resid = np.asarray(model.residuals, dtype=float)
        resid = resid[~np.isnan(resid)]
        n = len(resid)
        if n < 5:
            raise ValueError("Poucos resíduos para diagnóstico.")

        lags = min(self.lags, max(1, n // 4))

        # Ljung-Box no último lag apenas
        lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_pvalue = float(lb["lb_pvalue"].iloc[0])

        # ACF dos resíduos
        acf_vals = acf(resid, nlags=lags, fft=True)

        diagnostics = {
            "n_residuals": n,
            "resid_mean": float(np.mean(resid)),
            "resid_var": float(np.var(resid, ddof=1)),
            "lb_stat": lb_stat,
            "lb_pvalue": lb_pvalue,
            "acf": acf_vals.tolist(),
        }

        return diagnostics

class WalkForwardDiagnostics(Diagnostics):
    """
    Realiza diagnósticos em janelas móveis (walk-forward) de uma série temporal.
    """

    def __init__(self, min_train_size: int, horizon:int=1, step_size: int=1):
        self.min_train_size = min_train_size
        self.horizon = horizon
        self.step_size = step_size

    def diagnose(self, model: TimeSeriesModel, y: np.ndarray, fit_params: dict = {}, predict_params: dict = {}, **kwargs) -> Dict[str, Any]:
        from .evaluator import TimeSeriesEvaluator
        n = len(y)
        y_preds = []
        y_trues = []

        for end in range(self.min_train_size, n - self.horizon + 1, self.step_size):
            y_train = y[:end]
            y_true = y[end:end + self.horizon]
            adjusted_fit_params = fit_params.copy()
            if "exog" in fit_params.keys():
                adjusted_fit_params["exog"] = fit_params["exog"][:end]
            
            model.reset_model()
            model.fit(y_train, **adjusted_fit_params)
            
            adjusted_predict_params = predict_params.copy()
            if "exog" in predict_params.keys():
                adjusted_predict_params["exog"] = predict_params["exog"][end:end + self.horizon]
            y_pred = model.predict(steps=self.horizon, **adjusted_predict_params)
            y_preds.append(y_pred[-1]) # último passo da previsão
            y_trues.append(y_true[-1]) # último valor verdadeiro
            
        evaluator = TimeSeriesEvaluator()
        metrics = evaluator.evaluate(
            np.array(y_trues),
            np.array(y_preds),
            y_train=None,
            seasonal_periods=None
        )
        result = {}
        for key, value in metrics.items():
            if value is not None and value is not np.nan:
                result[key] = float(value)
        return result