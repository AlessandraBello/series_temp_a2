from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.data_loader import CSVDataLoader
from model.baseline_models import NaiveForecast, MovingAverage
from model.exp_smooth_models import SimpleExponentialSmoothingModel
from model.visualizer import TimeSeriesVisualizer
from model.evaluator import TimeSeriesEvaluator, TransformEvaluator
from model.diagnostics import ResidualDiagnostics
from model.preprocessor import LogDiffTransform, Differencer
from model.ar_ma_models import ARModel, MAModel


plt.style.use("seaborn-v0_8")


def train_test_split(y: np.ndarray, test_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide a série em treino e teste usando o último 'test_size' pontos como teste.
    """
    if test_size <= 0 or test_size >= len(y):
        raise ValueError("test_size deve ser > 0 e < len(y)")
    return y[:-test_size], y[-test_size:]


def evaluate_transformations_for_series(
    df: pd.DataFrame,
    series_names: list[str],
) -> pd.DataFrame:
    """
    Usa TimeSeriesEvaluator (ADF) para avaliar diferentes transformações
    (sem transformação, diferença de primeira ordem, log-diferença)
    para cada série.
    """
    evaluator = TransformEvaluator()
    results: list[Dict] = []

    for series_name in series_names:
        s = df[series_name].astype(float)

        transforms: Dict[str, pd.Series] = {
            "Original": s,
            "Diferença de primeira ordem": s.diff(),
        }

        # Log-diferença apenas se a série for estritamente positiva
        if (s > 0).all():
            log_diff = LogDiffTransform()
            diff_values = log_diff.transform(s.values)
            transforms["Diferença do log(x)"] = pd.Series(
                diff_values,
                index=s.index[1:],
            )

        for t_name, s_trans in transforms.items():
            res = evaluator.evaluate(
                series=s_trans,
                transform_name=t_name,
                series_name=series_name,
            )
            results.append(res)

    results_df = pd.DataFrame(results)
    print("\n=== Resultados ADF por transformação ===")
    print(
        results_df.pivot(
            index="series",
            columns="transform",
            values="p_value",
        )
    )

    best_per_series = (
        results_df.sort_values("p_value").groupby("series").first().reset_index()
    )

    print("\n=== Melhor transformação (menor p-valor ADF) por série ===")
    print(best_per_series)

    return results_df


def fit_baseline_models(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Ajusta modelos baseline em nível:
    - Naive
    - MovingAverage(3)
    - Simple Exponential Smoothing

    Retorna as previsões fora da amostra (tamanho len(y_test)).
    """
    forecasts: Dict[str, np.ndarray] = {}

    # Naive
    naive = NaiveForecast()
    naive.fit(y_train)
    forecasts["Naive"] = naive.predict(steps=len(y_test))

    # Média móvel (janela 3)
    ma3 = MovingAverage(window=3)
    ma3.fit(y_train)
    forecasts["MovingAverage(3)"] = ma3.predict(steps=len(y_test))

    # SES
    ses = SimpleExponentialSmoothingModel()
    ses.fit(y_train)
    forecasts["SimpleExpSmoothing"] = ses.predict(steps=len(y_test))

    return forecasts


def fit_ar_ma_models_on_diff(
    y_train: np.ndarray,
    y_test: np.ndarray,
    ar_lags: int = 2,
    ma_order: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Aplica diferenciação de primeira ordem e ajusta:

    - AR(ar_lags)
    - MA(ma_order)

    sobre a série diferenciada. Depois reconstrói previsões em nível
    usando o Differencer.inverse_transform.
    """
    forecasts: Dict[str, np.ndarray] = {}

    # AR(p) sobre série diferenciada
    diff_ar = Differencer(order=1)
    y_train_diff = diff_ar.transform(y_train)
    ar_model = ARModel(lags=ar_lags)
    ar_model.fit(y_train_diff)
    y_diff_forecast_ar = ar_model.predict(steps=len(y_test))
    y_pred_ar = diff_ar.inverse_transform(y_diff_forecast_ar, initial_value=y_train[-1])[1:]  # descarta o primeiro ponto extra
    forecasts[f"AR({ar_lags}) diff"] = y_pred_ar

    # MA(q) sobre série diferenciada
    diff_ma = Differencer(order=1)
    y_train_diff_ma = diff_ma.transform(y_train)
    ma_model = MAModel(order=ma_order)
    ma_model.fit(y_train_diff_ma)
    y_diff_forecast_ma = ma_model.predict(steps=len(y_test))
    y_pred_ma = diff_ma.inverse_transform(y_diff_forecast_ma, initial_value=y_train[-1])[1:]  # descarta o primeiro ponto extra
    forecasts[f"MA({ma_order}) diff"] = y_pred_ma

    return forecasts


def evaluate_forecasts(
    y_train: np.ndarray,
    y_test: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    seasonal_periods: int | None = None,
) -> pd.DataFrame:
    """
    Avalia as previsões usando TimeSeriesEvaluator.evaluate.
    """
    evaluator = TimeSeriesEvaluator()
    rows = []

    for name, y_pred in forecasts.items():
        metrics = evaluator.evaluate(
            y_true=y_test,
            y_pred=y_pred,
            y_train=y_train,
            seasonal_periods=seasonal_periods,
        )
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    results_df = pd.DataFrame(rows).set_index("model")
    print("\n=== Métricas de desempenho (teste) ===")
    print(results_df)

    return results_df


def diagnostics_for_diff_model(y_train: np.ndarray, lags: int = 24) -> None:
    """
    Exemplo: diagnósticos de resíduos da série diferenciada ajustada com AR(p).
    Aqui a ideia é mostrar como usar ResidualDiagnostics com um modelo em diff.
    """
    diff = Differencer(order=1)
    y_train_diff = diff.transform(y_train)

    ar_model = ARModel(lags=2)
    ar_model.fit(y_train_diff)

    diag = ResidualDiagnostics(lags=lags)
    res = diag.diagnose(ar_model, y_train_diff)

    print("\n=== Diagnóstico de resíduos para AR(2) em série diferenciada ===")
    print(f"n_residuals  : {res['n_residuals']}")
    print(f"mean(resid)  : {res['resid_mean']:.4f}")
    print(f"var(resid)   : {res['resid_var']:.4f}")
    print(f"Ljung-Box({lags}): stat = {res['lb_stat']:.4f}, p-valor = {res['lb_pvalue']:.4f}")
    print("  (p grande -> não rejeita hipótese de resíduos ~ ruído branco.)")


def main():
    # 1. Carregar dados
    data_path = "data/data_updated.csv"
    series_names = ["volume", "inv", "users"]  # ajuste conforme colunas no CSV
    target_column = "volume"
    test_horizon = 12

    loader = CSVDataLoader(
        file_path=data_path, date_column="week", target_column=target_column
    )
    df = loader.load()

    print(
        f"Dados carregados com shape {df.shape} e índice de datas baseado em 'week'."
    )

    # Plot das séries
    fig, axes = plt.subplots(len(series_names), 1, figsize=(10, 8), sharex=True)
    for ax, name in zip(axes, series_names):
        ax.plot(df[name].astype(float), label=f"{name} semanal")
        ax.set_ylabel(name)
        ax.legend(loc="upper left")
    axes[0].set_title("Séries originais")
    plt.tight_layout()
    plt.show()

    # Avalia transformações (ADF) por série
    _ = evaluate_transformations_for_series(df, series_names=series_names)

    # Foca em uma série para previsão
    y = df[target_column].astype(float).values
    y_train, y_test = train_test_split(y, test_size=test_horizon)
    print(f"\nSérie '{target_column}' – treino: {len(y_train)}, teste: {len(y_test)}")

    viz = TimeSeriesVisualizer(figsize=(12, 4))
    viz.plot(
        y,
        title=f"Série original - {target_column}",
        xlabel="Tempo",
        ylabel=target_column,
        label=target_column,
    )

    # Ajusta modelos baseline em nível
    baseline_forecasts = fit_baseline_models(y_train, y_test)

    # Ajusta modelos AR e MA sobre a série diferenciada
    ar_ma_forecasts = fit_ar_ma_models_on_diff(
        y_train=y_train,
        y_test=y_test,
        ar_lags=2,
        ma_order=1,
    )

    # Combina previsões
    all_forecasts = {**baseline_forecasts, **ar_ma_forecasts}

    results_df = evaluate_forecasts(
        y_train=y_train,
        y_test=y_test,
        forecasts=all_forecasts,
        seasonal_periods=None,
    )

    diagnostics_for_diff_model(y_train=y_train, lags=24)

    viz.plot_multiple_forecasts(
        y_train=y_train,
        forecasts=all_forecasts,
        y_true=y_test,
        title=f"Previsões de todos os modelos - {target_column}",
        xlabel="Tempo",
        ylabel=target_column,
    )


if __name__ == "__main__":
    main()
