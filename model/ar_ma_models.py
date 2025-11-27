"""
ar_ma_models.py

Modelos Autorregressivos (AR) e de Média Móvel (MA) para séries temporais,
seguindo o conteúdo da nota 10-11

Estacionariedade (AR): todas as raízes do polinômio ficam fora do círculo unitário.

Invertibilidade (MA): todas as raízes do polinômio ficam fora do círculo unitário.
ficam fora do círculo unitário.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import warnings

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from .base import TimeSeriesModel


def _check_roots_outside_unit_circle(coeffs: np.ndarray, poly_type: str) -> bool:
    """
    Verifica se todas as raízes do polinômio estão fora do círculo unitário.
    Parameters
    ----------
    coeffs : np.ndarray
    poly_type : {"ar", "ma"}
        Tipo do polinômio

    Returns
    -------
    bool
        True se todas as raízes têm módulo > 1.
    """
    coeffs = np.asarray(coeffs, dtype=float)

    if coeffs.size == 0:
        return True  
    if poly_type == "ar":
        poly = np.r_[1.0, -coeffs]
    elif poly_type == "ma":
        poly = np.r_[1.0, coeffs]
    else:
        raise ValueError("poly_type deve ser 'ar' ou 'ma'")

    roots = np.roots(poly)
    if roots.size == 0:
        return True

    return np.all(np.abs(roots) > 1.0)


class ARModel(TimeSeriesModel):
    """
    Modelo Autorregressivo AR(p):
    Implementado via statsmodels.tsa.ar_model.AutoReg.

    Este modelo deve ser aplicado em séries (ou transformações/diferenças da série)
    que sejam aproximadamente estacionárias.
    """

    def __init__(self, lags: int = 1, include_const: bool = True, name: str | None = None):
        model_name = name or f"AR({lags})"
        super().__init__(model_name)
        self.lags = lags
        self.include_const = include_const
        self._model = None
        self._fitted_model = None

        self.ar_params_: Optional[np.ndarray] = None
        self.is_stationary: Optional[bool] = None

    def fit(self, y: np.ndarray, **kwargs) -> "ARModel":
        """
        Ajusta o modelo AR(p) à série y.

        Parameters
        ----------
        y : np.ndarray
            Série temporal (idealmente estacionária).
        **kwargs :
            Passado para AutoReg.fit (p.ex., method="yule_walker" ou padrão "cmle").

        Returns
        -------
        ARModel
        """
        y = np.asarray(y, dtype=float)
        if len(y) <= self.lags:
            raise ValueError(f"Need at least {self.lags + 1} observations for AR({self.lags}).")

        trend = "c" if self.include_const else "n"
        self._model = AutoReg(y, lags=self.lags, trend=trend, old_names=False)
        self._fitted_model = self._model.fit(**kwargs)

        # Valores ajustados e resíduos
        self.fitted_values = np.asarray(self._fitted_model.fittedvalues, dtype=float)
        self.residuals = np.asarray(self._fitted_model.resid, dtype=float)
        self.y_train = y
        self.is_fitted = True

        # Coeficientes AR (ignorando intercepto se houver)
        params = np.asarray(self._fitted_model.params, dtype=float)
        if self.include_const:
            self.ar_params_ = params[1:]
        else:
            self.ar_params_ = params

        # Checar estacionariedade
        self.is_stationary = _check_roots_outside_unit_circle(self.ar_params_, poly_type="ar")
        if not self.is_stationary:
            warnings.warn(
                f"{self.name}: condições de estacionariedade podem não ser satisfeitas "
                f"(raízes do polinômio AR dentro ou sobre o círculo unitário).",
                UserWarning,
            )

        return self

    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Previsão h passos à frente.

        Parameters
        ----------
        steps : int
            Horizonte de previsão.

        Returns
        -------
        np.ndarray
            Previsões y_{T+1}, ..., y_{T+steps}
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        start = len(self.y_train)
        end = start + steps - 1
        forecast = self._fitted_model.predict(start=start, end=end)
        return np.asarray(forecast, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        if not self.is_fitted:
            return {}
        params = np.asarray(self._fitted_model.params, dtype=float)
        const = float(params[0]) if self.include_const and params.size > 0 else 0.0
        return {
            "ar_params": self.ar_params_,
            "const": const,
            "sigma2": float(self._fitted_model.sigma2),
            "aic": float(self._fitted_model.aic),
            "bic": float(self._fitted_model.bic),
            "is_stationary": self.is_stationary,
        }


class MAModel(TimeSeriesModel):
    """
    Modelo de Média Móvel MA(q):

        y_t = C + ε_t + θ_1 ε_{t-1} + ... + θ_q ε_{t-q},  ε_t ~ WN(0, σ^2)

    Implementado via statsmodels.tsa.arima.model.ARIMA com order=(0, 0, q).

    Este modelo deve ser aplicado em séries estacionárias.
    """

    def __init__(self, order: int = 1, include_const: bool = True, name: str | None = None):
        model_name = name or f"MA({order})"
        super().__init__(model_name)
        self.order = order
        self.include_const = include_const
        self._model = None
        self._fitted_model = None

        self.ma_params_: Optional[np.ndarray] = None
        self.is_invertible: Optional[bool] = None

    def fit(self, y: np.ndarray, **kwargs) -> "MAModel":
        """
        Ajusta o modelo MA(q) à série y.

        Parameters
        ----------
        y : np.ndarray
            Série temporal (idealmente estacionária).
        **kwargs :
            Passado para ARIMA.fit.

        Returns
        -------
        MAModel
        """
        y = np.asarray(y, dtype=float)
        if len(y) <= self.order:
            raise ValueError(f"Need at least {self.order + 1} observations for MA({self.order}).")

        trend = "c" if self.include_const else "n"
        self._model = ARIMA(y, order=(0, 0, self.order), trend=trend)
        self._fitted_model = self._model.fit(**kwargs)

        self.fitted_values = np.asarray(self._fitted_model.fittedvalues, dtype=float)
        self.residuals = np.asarray(self._fitted_model.resid, dtype=float)
        self.y_train = y
        self.is_fitted = True

        # Coeficientes MA
        if hasattr(self._fitted_model, "maparams"):
            self.ma_params_ = np.asarray(self._fitted_model.maparams, dtype=float)
        else:
            params = self._fitted_model.params
            if hasattr(params, "index"):
                names = list(params.index)
                ma_coeffs = [
                    coef
                    for coef, name in zip(params.values, names)
                    if name.lower().startswith("ma")
                ]
                self.ma_params_ = np.asarray(ma_coeffs, dtype=float)
            else:   
                params_arr = np.asarray(params, dtype=float)
                self.ma_params_ = params_arr[-self.order :]

        # Checar invertibilidade
        self.is_invertible = _check_roots_outside_unit_circle(self.ma_params_, poly_type="ma")
        if not self.is_invertible:
            warnings.warn(
                f"{self.name}: condições de invertibilidade podem não ser satisfeitas "
                f"(raízes do polinômio MA dentro ou sobre o círculo unitário).",
                UserWarning,
            )

        return self

    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Previsão h passos à frente.

        Parameters
        ----------
        steps : int
            Horizonte de previsão.

        Returns
        -------
        np.ndarray
            Previsões y_{T+1}, ..., y_{T+steps}
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        forecast = self._fitted_model.forecast(steps)
        return np.asarray(forecast, dtype=float)

    def get_params(self) -> Dict[str, Any]:
        if not self.is_fitted:
            return {}
        params = np.asarray(self._fitted_model.params, dtype=float)
        const = float(params[0]) if self.include_const and params.size > 0 else 0.0
        return {
            "ma_params": self.ma_params_,
            "const": const,
            "sigma2": float(self._fitted_model.sigma2),
            "aic": float(self._fitted_model.aic),
            "bic": float(self._fitted_model.bic),
            "is_invertible": self.is_invertible,
        }
