"""
Módulo para preprocessamento de séries temporais.
Transformações como log, diferenciação, normalização, etc.
"""
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

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Reconstrói a série original a partir das diferenças do log.

        Note que a transformação de diferença perde uma observação; para
        permitir a reconstrução, utilizamos o primeiro valor em escala de log
        armazenado durante `transform`.
        """
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            return y

        if self._initial_log_value is None:
            raise ValueError(
                "LogDiffTransform.inverse_transform foi chamado antes de "
                "transform, logo o valor inicial não está definido."
            )

        # Reconstrói o log via soma cumulativa das diferenças
        log_series = np.concatenate(
            [[self._initial_log_value], self._initial_log_value + np.cumsum(y)]
        )
        # Volta para a escala original
        return np.exp(log_series) - self.eps
