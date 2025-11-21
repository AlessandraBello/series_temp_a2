"""
Framework de Séries Temporais

Este pacote fornece uma estrutura modular e extensível para trabalhar com
séries temporais, incluindo carregamento de dados, preprocessamento, modelagem,
avaliação, diagnósticos e visualização.

Estrutura:
- base: Classes base abstratas que definem as interfaces
- data_loader: Carregamento de dados
- preprocessor: Transformações de séries temporais
- baseline_models: Implementações de modelos baseline
- exp_smooth_models: Implementação dos modelos de Suavização Exponencial.
- evaluator: Métricas de avaliação
- diagnostics: Análise de resíduos e diagnósticos
- visualizer: Visualização de séries e resultados
"""

# Classes base
from .base import (
    TimeSeriesModel,
    Preprocessor,
    Evaluator,
    Diagnostics,
    Visualizer,
)

# Data loading
from .data_loader import (
    CSVDataLoader,
)

# Preprocessing
from .preprocessor import (
    LogTransform,
)

# Models
from .baseline_models import (
    NaiveForecast,
    MovingAverage,
)

# Exponential Smoothing Models 
from .exp_smooth_models import (
    SimpleExponentialSmoothingModel,
    HoltLinearTrendModel,
    HoltWintersAdditiveModel,
    HoltWintersMultiplicativeModel
)

# Evaluation
from .evaluator import (
    TimeSeriesEvaluator,
)

# Diagnostics
from .diagnostics import (
    ResidualDiagnostics,
)

# Visualization
from .visualizer import (
    TimeSeriesVisualizer,
)

__all__ = [
    # Base classes
    'TimeSeriesModel',
    'Preprocessor',
    'Evaluator',
    'Diagnostics',
    'Visualizer',
    # Data loading
    'CSVDataLoader',
    # Preprocessing
    'LogTransform',
    # Models
    'NaiveForecast',
    'MovingAverage',
    # Evaluation
    'TimeSeriesEvaluator',
    # Diagnostics
    'ResidualDiagnostics',
    # Visualization
    'TimeSeriesVisualizer',
]
