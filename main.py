"""
main.py

Script de experimento para séries temporais:

1. Carrega dados de um CSV.
2. Testa estacionariedade (ADF + KPSS).
3. Aplica diferenciação de primeira ordem, se desejado.
4. Ajusta baselines simples (Naive, MovingAverage, SES).
5. Ajusta modelos AR(p) e MA(q) sobre a série estacionária (PDF 11).
6. Compara desempenho (MAE, RMSE, MAPE, MASE, RMSSE).
7. Faz diagnóstico dos resíduos (Ljung-Box, ACF, etc.).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss

# Imports do framework (assumindo que este arquivo está no mesmo pacote)
from .data_loader import CSVDataLoader
from .baseline_model import NaiveForecast, MovingAverage
from .exp_smooth_models import SimpleExponentialSmoothingModel
from .visualizer import TimeSeriesVisualizer
from .evaluator import TimeSeriesEvaluator
from .diagnostics import ResidualDiagnostics
from .preprocessors import Differencer
from .ar_ma_models import ARModel, MAModel

plt.style.use("seaborn-v0_8")


def stationarity_tests(series: np.ndarray, series_name: str = "y") -> None:
    """
    Roda ADF e KPSS na série e imprime um pequeno relatório.

    ADF: H0 = série tem raiz unitária (não estacionária)
    KPSS: H0 = série é estacionária
    """
    clean = pd.Series(series).dropna().values

    # ADF
    adf_stat, adf_p, _, _ = adfuller(clean)
    # KPSS (com lag automático)
    kpss_stat, kpss_p, _, _ = kpss(clean, nlags="auto")

    print(f"\n=== Testes de estacionariedade para {series_name} ===")
    print(f"ADF:  estatística = {adf_stat:.4f}, p-valor = {adf_p:.4f}")
    print(f"KPSS: estatística = {kpss_stat:.4f}, p-valor = {kpss_p:.4f}")
    print("Interpretação rápida:")
    print("  - ADF: p pequeno -> rejeita H0 de raiz unitária (p.ex. < 0.05 sugere estacionária).")
    print("  - KPSS: p pequeno -> rejeita H0 de estacionariedade (p.ex. < 0.05 sugere não estacionária).")


def train_test_split(y: np.ndarray, test_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide a série em treino e teste usando o último 'test_size' pontos como teste.
    """
    if test_size <= 0 or test_size >= len(y):
        raise ValueError("test_size deve ser > 0 e < len(y)")
    return y[:-test_size], y[-test_size:]


def fit_and_forecast_level_model(model, y_train: np.ndarray, y_test: np.ndarray, name: str):
    """
    Ajusta um modelo em nível (sem diferenciação) e retorna previsões + erro.
    """
    print(f"\n=== Ajustando modelo em nível: {name} ===")
    model.fit(y_train)
    y_pred = model.predict(steps=len(y_test))
    return y_pred


def fit_and_forecast_diff_model(model, differencer: Differencer,
                                y_train: np.ndarray, y_test: np.ndarray, name: str):
    """
    Ajusta um modelo AR/MA na série diferenciada e reconstrói previsão em nível.

    Passos:
    - Aplica diferença em y_train.
    - Ajusta modelo na série estacionária.
    - Prevê diferenças futuras.
    - Usa o Differencer.inverse_transform para voltar à escala de nível.
    """
    print(f"\n=== Ajustando modelo em série diferenciada: {name} ===")

    # Ajuste na série diferenciada
    y_train_diff = differencer.transform(y_train) 
    model.fit(y_train_diff)

    # Previsão da série diferenciada
    y_diff_forecast = model.predict(steps=len(y_test))

    # Reconstrução da previsão em nível (retorna a série original)
    y_level_forecast = differencer.inverse_transform(y_diff_forecast)
    return y_level_forecast


def evaluate_models(y_train: np.ndarray,
                    y_test: np.ndarray,
                    forecasts: dict[str, np.ndarray],
                    seasonal_periods: int | None = None) -> None:
    """
    Avalia vários modelos usando TimeSeriesEvaluator.
    """
    evaluator = TimeSeriesEvaluator(y_train=y_train, seasonal_periods=seasonal_periods)

    print("\n=== Métricas de desempenho (teste) ===")
    rows = []
    for name, y_pred in forecasts.items():
        metrics = evaluator.evaluate(y_true=y_test, y_pred=y_pred)
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    print(results_df.set_index("model"))


def diagnostics_for_model(model, y_train: np.ndarray, name: str) -> None:
    """
    Roda diagnóstico de resíduos (Ljung-Box, etc.) e imprime um resumo.
    """
    diag = ResidualDiagnostics(lags=24)
    res = diag.diagnose(model, y_train)

    print(f"\n=== Diagnóstico de resíduos: {name} ===")
    print(f"n_residuals  : {res['n_residuals']}")
    print(f"mean(resid)  : {res['resid_mean']:.4f}")
    print(f"var(resid)   : {res['resid_var']:.4f}")
    print(f"Ljung-Box({diag.lags}): stat = {res['lb_stat']:.4f}, p-valor = {res['lb_pvalue']:.4f}")
    print("  (p grande -> não rejeita hipótese de resíduos ~ ruído branco.)")


def main():
    # 1. Carregar dados
    data_path = "data/data_updated.csv"
    target_column = "volume" 
    test_horizon = 12         

    loader = CSVDataLoader(file_path=data_path, date_column="week", target_column=target_column)
    y = loader.get_series()  

    print(f"Série '{target_column}' carregada com {len(y)} observações.")

    # Plot da série
    viz = TimeSeriesVisualizer(figsize=(10, 4))
    viz.plot(y, title=f"Série original - {target_column}",
             xlabel="Tempo", ylabel=target_column, label=target_column)

    # Dividir em treino e teste
    y_train, y_test = train_test_split(y, test_size=test_horizon)
    print(f"Tamanho treino: {len(y_train)}, teste: {len(y_test)}")

    # Testes de estacionariedade na série em nível
    stationarity_tests(y_train, series_name=f"{target_column} (treino)")

    # Ajustar baselines simples em nível
    naive = NaiveForecast()
    ma3 = MovingAverage(window=3)
    ses = SimpleExponentialSmoothingModel()

    y_pred_naive = fit_and_forecast_level_model(naive, y_train, y_test, "Naive")
    y_pred_ma3 = fit_and_forecast_level_model(ma3, y_train, y_test, "MovingAverage(3)")
    y_pred_ses = fit_and_forecast_level_model(ses, y_train, y_test, "SimpleExpSmoothing")

    # Diferenciação + modelos AR e MA
    # Diferenciar uma vez (remove tendência linear).
    # Checar ADF/KPSS após a diferença para confirmar estacionariedade.
    differencer = Differencer(order=1)

    # Modelo AR(p)
    ar_model = ARModel(lags=2)  
    y_pred_ar = fit_and_forecast_diff_model(
        ar_model, differencer, y_train, y_test, name="AR(2) sobre série diferenciada"
    )

    # Modelo MA(q)
    ma_model = MAModel(order=1)  
    # Re-inicializar o Differencer (pois ele guarda o último valor do treino)
    differencer_ma = Differencer(order=1)
    y_pred_ma = fit_and_forecast_diff_model(
        ma_model, differencer_ma, y_train, y_test, name="MA(1) sobre série diferenciada"
    )

    # Avaliar todos os modelos
    forecasts = {
        "Naive": y_pred_naive,
        "MovingAverage(3)": y_pred_ma3,
        "SimpleExpSmoothing": y_pred_ses,
        "AR(2) diff": y_pred_ar,
        "MA(1) diff": y_pred_ma,
    }

    evaluate_models(y_train=y_train, y_test=y_test, forecasts=forecasts, seasonal_periods=None)

    # Diagnósticos de resíduos para alguns modelos
    diagnostics_for_model(naive, y_train, "Naive (nível)")
    diagnostics_for_model(ar_model, y_train_diff := Differencer(order=1).transform(y_train),
                          "AR(2) (série diferenciada)")

    # Visualização: treino + previsões
    viz.plot_forecast(
        y_train=y_train,
        y_forecast=y_pred_ar,
        y_true=y_test,
        title=f"Previsão AR(2) sobre série diferenciada (reconstruída) - {target_column}",
        xlabel="Tempo",
        ylabel=target_column,
    )


if __name__ == "__main__":
    main()
