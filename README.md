# series_temp_a2

Trabalho de graduação em análise e previsão de séries temporais.

## Sobre o Trabalho

Este projeto apresenta uma análise de séries temporais, com foco na modelagem, avaliação e diagnóstico de diferentes métodos de previsão. Os resultados e análises estão documentados em notebooks Jupyter, que constituem o produto principal deste trabalho.

## Notebooks Principais

Os notebooks contêm todas as análises, resultados e discussões do trabalho:

- **`main.ipynb`**: Análise principal com modelagem, comparação de modelos e resultados
- **`model_diagnostics.ipynb`**: Diagnósticos detalhados dos modelos (análise de resíduos, testes estatísticos)
- **`exog_example.ipynb`**: Exemplos de uso de variáveis exógenas nos modelos

## Framework de Suporte

Para suportar as análises realizadas nos notebooks, foi desenvolvido um framework modular em Python organizado em:

- **`base`**: Classes abstratas que definem interfaces padronizadas
- **`data_loader`**: Carregamento e leitura de dados CSV
- **`preprocessor`**: Transformações de séries temporais
- **`baseline_models`**: Modelos baseline (Naive, Média Móvel)
- **`exp_smooth_models`**: Modelos de Suavização Exponencial (Simple, Holt, Holt-Winters)
- **`evaluator`**: Métricas de avaliação de modelos
- **`diagnostics`**: Análise de resíduos e diagnósticos estatísticos
- **`visualizer`**: Visualizações de séries temporais e resultados

## Instalação

```bash
pip install -r requirements.txt
```

## Como Executar

## Dados

Os dados utilizados estão em `data/data_updated.csv` e contêm séries temporais semanais com variáveis de volume, investimento e usuários.

## Modelos

Os modelos utilizados são:

- Modelo Naive
- Modelo de Média Móvel
- Modelo de Suavização Exponencial
- Modelo de Suavização Exponencial de Holt
- Modelo de Suavização Exponencial de Holt-Winters
- Modelo AR
- Modelo MA