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
