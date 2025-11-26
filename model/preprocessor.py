"""
Transformações para séries temporais.

Aqui focamos em:
- Differencer(order=1): diferenciação de primeira ordem

Conforme o PDF 11, a diferenciação de primeira ordem:

    y'_t = y_t - y_{t-1}

remove tendência linear e ajuda a tornar a série estacionária.

"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Preprocessor


class Differencer(Preprocessor):
    """
    Diferenciação de primeira ordem (order=1).

    Uso típico:
    - Aplicar em y para obter uma série mais próxima de estacionária.
    - Ajustar modelos AR/MA em y_dif.
    - Para previsões, prever diferenças futuras e depois "reintegration"
      usando o último valor observado da série original.
    """

    def __init__(self, order: int = 1):
        if order not in (0, 1):
            raise NotImplementedError("Differencer atualmente implementa apenas order=0 ou 1.")
        self.order = order
        self._last_train_value: Optional[float] = None

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Aplica a diferenciação de primeira ordem na série.

        Parameters
        ----------
        y : np.ndarray
            Série em nível.

        Returns
        -------
        np.ndarray
            Série diferenciada (length T-1 se order=1).
        """
        y = np.asarray(y, dtype=float)

        if self.order == 0:
            self._last_train_value = y[-1]
            return y

        if len(y) < 2:
            raise ValueError("Need at least 2 observations to difference.")

        self._last_train_value = float(y[-1])
        return np.diff(y, n=1)

    def inverse_transform(self, y_diff_forecast: np.ndarray) -> np.ndarray:
        """
        Reconstrói previsões em nível a partir de previsões na escala diferenciada.

        Suponha:
        - y_T é o último valor observado da série original (armazenado durante o transform).
        - y_diff_forecast contém previsões de y'_{T+1}, y'_{T+2}, ...

        Então:
        - y_{T+1} = y_T + y'_{T+1}
        - y_{T+2} = y_{T+1} + y'_{T+2}
        - ...

        Parameters
        ----------
        y_diff_forecast : np.ndarray
            Previsões na escala diferenciada.

        Returns
        -------
        np.ndarray
            Previsões na escala original (nível).
        """
        if self.order == 0:
            return np.asarray(y_diff_forecast, dtype=float)

        if self._last_train_value is None:
            raise ValueError("Differencer needs to be fitted (transform called) before inverse_transform.")

        y_diff_forecast = np.asarray(y_diff_forecast, dtype=float)
        y_level = np.empty_like(y_diff_forecast)

        prev = self._last_train_value
        for i, d in enumerate(y_diff_forecast):
            prev = prev + d
            y_level[i] = prev

        return y_level
