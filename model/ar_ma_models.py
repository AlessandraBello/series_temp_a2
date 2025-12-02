
"""
ar_ma_models.py

Modelos Autorregressivos (AR) e de Média Móvel (MA) para séries temporais,
seguindo o conteúdo da nota 10-11

Estacionariedade (AR): todas as raízes do polinômio ficam fora do círculo unitário.

Invertibilidade (MA): todas as raízes do polinômio ficam fora do círculo unitário.
"""
from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import warnings

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from .base import TimeSeriesModel

from abc import ABC, abstractmethod

class ARMAModel(TimeSeriesModel):
    
    @abstractmethod
    def fit(self, y: np.ndarray, **kwargs) -> "ARMAModel":
        pass
    
    @abstractmethod
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        pass
    
    
    def _check_roots_outside_unit_circle(
        self,
        coeffs: np.ndarray,
        poly_type: str,
        tol: float = 1e-3,
    ) -> tuple[bool, np.ndarray]:
        """
        Verifica se todas as raízes do polinômio estão fora do círculo unitário.

        Parameters
        ----------
        coeffs : np.ndarray
            Coeficientes do polinômio (AR ou MA).
        poly_type : {"ar", "ma"}
            Tipo do polinômio.
        tol : float
            Tolerância numérica. Consideramos "problema" se |raiz| <= 1 + tol.

        Returns
        -------
        (bool, np.ndarray)
            - bool: True se todas as raízes têm módulo > 1 + tol.
            - np.ndarray: array de raízes.
        """
        coeffs = np.asarray(coeffs, dtype=float)

        if coeffs.size == 0:
            # Sem coeficientes => não temos raiz relevante
            return True, np.array([], dtype=float)

        if poly_type == "ar":
            poly = np.r_[-coeffs[::-1], 1.0]

        elif poly_type == "ma":
            poly = np.r_[coeffs[::-1], 1.0]
        else:
            raise ValueError("poly_type deve ser 'ar' ou 'ma'")

        roots = np.roots(poly)
        if roots.size == 0:
            return True, roots

        mod = np.abs(roots)
        is_outside = np.all(mod > 1.0 + tol)
        return is_outside, roots
    
    def reset_model(self):
        super().reset_model()
        self._model = None
        self._fitted_model = None

        self.params_: Optional[np.ndarray] = None
        self.is_stationary: Optional[bool] = None
        self.ar_roots_: Optional[np.ndarray] = None
        self.ar_roots_mod_: Optional[np.ndarray] = None
    
    def get_params(self) -> Dict[str, Any]:
        if not self.is_fitted:
            return {}
        params = np.asarray(self._fitted_model.params, dtype=float)
        const = float(params[0]) if self.include_const and params.size > 0 else 0.0
        return {
            "params": self.params_,
            "const": const,
            "aic": float(self._fitted_model.aic),
            "bic": float(self._fitted_model.bic)
        }



class ARModel(ARMAModel):
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

        self.params_: Optional[np.ndarray] = None
        self.is_stationary: Optional[bool] = None
        self.ar_roots_: Optional[np.ndarray] = None
        self.ar_roots_mod_: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray, verbose=False, **kwargs) -> "ARModel":
        """
        Ajusta o modelo AR(p) à série y.

        Parameters
        ----------
        y : np.ndarray
            Série temporal (idealmente estacionária).
        **kwargs :
            Passado para AutoReg.fit

        Returns
        -------
        ARModel
        """
        if not verbose:
            print = lambda *args, **kwargs: None  # noqa: E731
            
        y = np.asarray(y, dtype=float)
        if len(y) <= self.lags:
            raise ValueError(f"Need at least {self.lags + 1} observations for AR({self.lags}).")

        exog_data = kwargs.pop("exog", None)
        
        trend = "c" if self.include_const else "n"
        self._model = AutoReg(y, lags=self.lags, trend=trend, old_names=False, exog=exog_data)
        self._fitted_model = self._model.fit(**kwargs)

        self.fitted_values = np.asarray(self._fitted_model.fittedvalues, dtype=float)
        self.residuals = np.asarray(self._fitted_model.resid, dtype=float)
        self.y_train = y
        self.is_fitted = True

        params = np.asarray(self._fitted_model.params, dtype=float)
        if self.include_const:
            self.params_ = params[1:]
            const = params[0]
        else:
            self.params_ = params
            const = 0.0

        print(f"[ARModel] {self.name} - constante (intercepto): {const}")
        print(f"[ARModel] {self.name} - parâmetros AR: {self.params_}")

        self.is_stationary, roots = self._check_roots_outside_unit_circle(
            self.params_,
            poly_type="ar",
        )
        self.ar_roots_ = roots
        self.ar_roots_mod_ = np.abs(roots)

        if roots.size > 0:
            print(f"[ARModel] {self.name} - raízes do polinômio AR: {roots}")
            print(f"[ARModel] {self.name} - |raízes|: {self.ar_roots_mod_}")
        else:
            print(f"[ARModel] {self.name} - nenhum root calculado (coeficientes vazios).")

        if not self.is_stationary:
            warnings.warn(
                f"{self.name}: condições de estacionariedade podem não ser satisfeitas "
                f"(raízes do polinômio AR dentro ou sobre o círculo unitário com tolerância).",
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
        
        exog_forecast = kwargs.pop("exog", None)

        start = len(self.y_train)
        end = start + steps - 1
        forecast = self._fitted_model.predict(start=start, end=end, exog_oos=exog_forecast)
        return np.asarray(forecast, dtype=float)

class MAModel(ARMAModel):
    """
    Modelo de Média Móvel MA(q)

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

        self.params_: Optional[np.ndarray] = None
        self.is_invertible: Optional[bool] = None
        self.ma_roots_: Optional[np.ndarray] = None
        self.ma_roots_mod_: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray, verbose=False, **kwargs) -> "MAModel":
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
        if not verbose:
            print = lambda *args, **kwargs: None  # noqa: E731
        
        y = np.asarray(y, dtype=float)
        if len(y) <= self.order:
            raise ValueError(f"Need at least {self.order + 1} observations for MA({self.order}).")
        
        exog = kwargs.pop("exog", None)

        trend = "c" if self.include_const else "n"
        self._model = ARIMA(
            y,
            order=(0, 0, self.order),
            trend=trend,
            enforce_stationarity=False,   
            enforce_invertibility=True,   
            exog=exog,
        )
        self._fitted_model = self._model.fit(**kwargs)

        self.fitted_values = np.asarray(self._fitted_model.fittedvalues, dtype=float)
        self.residuals = np.asarray(self._fitted_model.resid, dtype=float)
        self.y_train = y
        self.is_fitted = True

        if hasattr(self._fitted_model, "maparams"):
            self.params_ = np.asarray(self._fitted_model.maparams, dtype=float)
        else:
            params = self._fitted_model.params
            if hasattr(params, "index"):
                names = list(params.index)
                ma_coeffs = [
                    coef
                    for coef, name in zip(params.values, names)
                    if name.lower().startswith("ma")
                ]
                self.params_ = np.asarray(ma_coeffs, dtype=float)
            else:
                params_arr = np.asarray(params, dtype=float)
                self.params_ = params_arr[-self.order :]

        # Constante (se houver)
        params_arr_full = np.asarray(self._fitted_model.params, dtype=float)
        const = float(params_arr_full[0]) if self.include_const and params_arr_full.size > 0 else 0.0

        # --- LOG: parâmetros MA ---
        print(f"[MAModel] {self.name} - constante (intercepto): {const}")
        print(f"[MAModel] {self.name} - parâmetros MA: {self.params_}")

        # Checar invertibilidade + obter raízes
        self.is_invertible, roots = self._check_roots_outside_unit_circle(
            self.params_,
            poly_type="ma",
        )
        self.ma_roots_ = roots
        self.ma_roots_mod_ = np.abs(roots)

        # --- LOG: raízes e módulos ---
        if roots.size > 0:
            print(f"[MAModel] {self.name} - raízes do polinômio MA: {roots}")
            print(f"[MAModel] {self.name} - |raízes|: {self.ma_roots_mod_}")
        else:
            print(f"[MAModel] {self.name} - nenhum root calculado (coeficientes vazios).")

        if not self.is_invertible:
            warnings.warn(
                f"{self.name}: condições de invertibilidade podem não ser satisfeitas "
                f"(raízes do polinômio MA dentro ou sobre o círculo unitário "
                f"com tolerância).",
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
        
        exog_forecast = kwargs.pop("exog", None)
        
        forecast = self._fitted_model.forecast(steps, exog=exog_forecast)
        return np.asarray(forecast, dtype=float)
