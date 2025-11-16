# Estrutura do Projeto - Séries Temporais

## Visão Geral

Este projeto fornece uma estrutura modular e extensível para trabalhar com séries temporais. A arquitetura é baseada em classes abstratas que definem interfaces claras, permitindo que diferentes implementações sejam facilmente adicionadas e testadas.

## Estrutura de Diretórios

```
series_temp_a2/
├── model/                    # Pacote principal
│   ├── __init__.py          # Exporta todas as classes principais
│   ├── base.py              # Classes base abstratas
│   ├── data_loader.py       # Carregamento e validação de dados
│   ├── preprocessor.py      # Transformações de séries temporais
│   ├── model.py             # Implementações de modelos
│   ├── evaluator.py         # Métricas de avaliação
│   ├── diagnostics.py       # Análise de resíduos
│   └── visualizer.py        # Visualizações
├── data/
│   └── data_updated.csv     # Dados do projeto
├── requirements.txt         # Dependências
└── README.md
```

## Componentes Principais

### 1. Classes Base (`base.py`)

Define as interfaces que todos os componentes devem seguir:

- **`TimeSeriesModel`**: Interface para modelos de séries temporais
  - `fit(y, **kwargs)`: Ajusta o modelo aos dados
  - `predict(steps, **kwargs)`: Gera previsões
  - `get_params()`: Retorna parâmetros do modelo

- **`Preprocessor`**: Interface para transformações
  - `transform(y)`: Aplica transformação
  - `inverse_transform(y)`: Aplica transformação inversa

- **`Evaluator`**: Interface para avaliação
  - `evaluate(y_true, y_pred)`: Calcula métricas

- **`Diagnostics`**: Interface para diagnósticos
  - `diagnose(model, y, **kwargs)`: Realiza diagnósticos

- **`Visualizer`**: Interface para visualização
  - `plot(y, **kwargs)`: Cria visualizações

### 2. Carregamento de Dados (`data_loader.py`)

- **`CSVDataLoader`**: Carrega dados de arquivos CSV
  - Suporta conversão automática de datas
  - Permite seleção de colunas específicas

### 3. Preprocessamento (`preprocessor.py`)

Transformações disponíveis:

- **`LogTransform`**: Transformação logarítmica (a ser implementado)

### 4. Modelos (`model.py`)

Implementações de exemplo:

- **`NaiveForecast`**: Modelo baseline (último valor)
- **`MovingAverage`**: Média móvel simples

**Nota**: Novos modelos devem herdar de `TimeSeriesModel` e implementar os métodos abstratos.

### 5. Avaliação (`evaluator.py`)

- **`TimeSeriesEvaluator`**: Calcula múltiplas métricas (a ser implementado)

### 6. Diagnósticos (`diagnostics.py`)

- **`ResidualDiagnostics`**: Análise de resíduos

### 7. Visualização (`visualizer.py`)

- **`TimeSeriesVisualizer`**: Cria visualizações
  - Plot de séries temporais
  - Plot de previsões vs. valores reais
  - Análise de resíduos (4 gráficos)
  - (conferir e testar)

## Extendendo o Framework

### Adicionar um Novo Modelo

```python
from model.base import TimeSeriesModel
import numpy as np

class MeuModelo(TimeSeriesModel):
    def __init__(self, param1=10):
        super().__init__("MeuModelo")
        self.param1 = param1
    
    def fit(self, y: np.ndarray, **kwargs) -> 'MeuModelo':
        # Implementar lógica de treinamento
        self.is_fitted = True
        self.fitted_values = ...  # Valores ajustados
        self.residuals = y - self.fitted_values
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        # Implementar lógica de previsão
        return np.array([...])
```

## Dependências

- `numpy`: Operações numéricas
- `pandas`: Manipulação de dados
- `matplotlib`: Visualizações
- `statsmodels`: Modelos estatísticos (para implementações futuras)

