"""
Módulo para diagnósticos de modelos de séries temporais.
Exemplo de diagnóstico: análise de resíduos
"""
import numpy as np
from .base import Diagnostics, TimeSeriesModel
from typing import Dict, Any

class ResidualDiagnostics(Diagnostics):
    def __init__(self):
        super().__init__()

    
    def diagnose(self, model: TimeSeriesModel, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Avalia a qualidade de previsões usando MAE, RMSE e MAPE.
        
        Parameters
        ----------
        model: TimeSeriesModel
            Modelo a ser diagnosticado
        y: np.ndarray
            Dados observados
        **kwargs
            Parâmetros adicionais (caso necessário)
            
        Returns
        -------
        Dict[str, Any]
            Dicionário com os resultados dos diagnósticos. Chaves são os nomes das métricas.
        """
        y_pred = model.predict(len(y))
        errors = y - y_pred
        return {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mape": float(np.mean(np.abs(errors / y)) * 100.0),
        }