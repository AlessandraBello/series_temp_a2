"""
Módulo para preprocessamento de séries temporais.
Transformações como log, diferença, normalização, etc.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from .base import Preprocessor


class LogTransform(Preprocessor):
    """
    Transformação logarítmica simples.

    Útil para estabilizar a variância em séries onde a amplitude das
    oscilações cresce com o nível da série (comportamento multiplicativo).
    """

    def __init__(self, eps: float = 1e-8):
        """
        Parameters
        ----------
        eps : float
            Pequena constante adicionada para evitar log(0).
        """
        self.eps = eps

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Aplica log aos dados.

        Assume que `y` é não negativo. Valores menores ou iguais a zero são
        deslocados por `eps` para permitir o cálculo do log.
        """
        y = np.asarray(y, dtype=float)
        return np.log(np.maximum(y, 0) + self.eps)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Aplica a transformação inversa do log.
        """
        y = np.asarray(y, dtype=float)
        return np.exp(y) - self.eps


class LogDiffTransform(Preprocessor):
    """
    Diferença de primeira ordem aplicada ao log da série.

    - O log estabiliza a variância.
    - A diferença no tempo remove tendência determinística remanescente.

    Esta transformação é útil para obter séries aproximadamente estacionárias
    a partir de séries com crescimento multiplicativo e tendência.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self._initial_log_value: float | None = None

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Aplica log e depois a diferença de primeira ordem.

        Guarda internamente o primeiro valor em escala de log para permitir
        uma inversão consistente via `inverse_transform`.
        """
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            return y

        log_y = np.log(np.maximum(y, 0) + self.eps)
        self._initial_log_value = float(log_y[0])
        return np.diff(log_y)

    def inverse_transform(self, y: np.ndarray, initial_log_value: float | None = None) -> np.ndarray:
        """
        Reconstrói a série original a partir das diferenças do log.

        Note que a transformação de diferença perde uma observação; para
        permitir a reconstrução, utilizamos o primeiro valor em escala de log
        armazenado durante `transform`.
        """
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            return y

        base = (
            initial_log_value
            if initial_log_value is not None
            else self._initial_log_value
        )
        if base is None:
            raise ValueError(
                "Nenhum initial_log_value foi fornecido e nenhum valor "
                "foi armazenado via transform()."
            )

        # Reconstrói o log via soma cumulativa das diferenças
        log_series = np.concatenate([[base], base + np.cumsum(y)])
        # Volta para a escala original
        return np.exp(log_series) - self.eps


class Differencer(Preprocessor):
    """
    Diferenciação de primeira ordem em nível.

    Esta transformação é exatamente a usada no conteúdo do 11-time_series.pdf
    para tornar a série mais próxima de estacionária:

        y'_t = y_t - y_{t-1}

    É esta transformação que vamos usar como base para ajustar modelos AR/MA
    (AR(p), MA(q)) na série diferenciada.
    """

    def __init__(self, order: int = 1):
        if order not in (0, 1):
            raise NotImplementedError(
                "Differencer atualmente implementa apenas order=0 ou 1."
            )
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
            self._last_train_value = float(y[-1])
            return y

        if len(y) < 2:
            raise ValueError("Need at least 2 observations to difference.")

        self._last_train_value = float(y[-1])
        return np.diff(y, n=1)

    def inverse_transform(self, y_diff_forecast: np.ndarray, initial_value: float | None = None) -> np.ndarray:
        """
        Reconstrói previsões em nível a partir de previsões na escala diferenciada.

        Suponha:
        - y_T é o último valor observado da série original (armazenado em transform ou passado como parâmetro).
        - y_diff_forecast contém previsões de y'_{T+1}, y'_{T+2}, ...

        Então:
        - y_{T+1} = y_T + y'_{T+1}
        - y_{T+2} = y_{T+1} + y'_{T+2}
        - ...

        Parameters
        ----------
        y_diff_forecast : np.ndarray
            Previsões na escala diferenciada.
        initial_value : float, optional
            Valor inicial y_T para reconstrução. Se None, usa o valor armazenado
            durante `transform`.

        Returns
        -------
        np.ndarray
            Previsões na escala original (nível).
        """
        
        if self.order == 0:
            return np.asarray(y_diff_forecast, dtype=float)

        if self._last_train_value is None:
            raise ValueError(
                "Differencer precisa ser ajustado (transform chamado) "
                "antes de inverse_transform."
            )

        if initial_value is not None:
            prev = initial_value
        else:
            if self._last_train_value is None:
                raise ValueError(
                    "Nenhum initial_value foi fornecido e nenhum valor "
                    "foi armazenado via transform()."
                )
            prev = self._last_train_value

        y_diff_forecast = np.asarray(y_diff_forecast, dtype=float)
        y_level = np.empty_like(y_diff_forecast)

        y_level = np.concatenate([[prev], prev + np.cumsum(y_diff_forecast)])

        return y_level