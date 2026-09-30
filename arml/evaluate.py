"""Laco de avaliacao comum aos experimentos, com ajuste, predicao, metricas e gravacao."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from arml.config import RESULTS, TABLES
from arml.metrics import bootstrap_ci, classification_report
from arml.splits import Split


def fit_score(pipeline_factory: Callable, X: pd.DataFrame, y: np.ndarray, split: Split) -> dict:
    """Ajusta no treino da particao e devolve a probabilidade prevista no teste."""
    modelo = pipeline_factory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(X.iloc[split.treino], y[split.treino])
        proba = modelo.predict_proba(X.iloc[split.teste])[:, 1]
    return {"modelo": modelo, "y_true": y[split.teste], "y_score": proba}


def evaluate_zoo(
    zoo: dict[str, Callable], X: pd.DataFrame, y: np.ndarray, split: Split
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Avalia cada modelo na mesma particao, com intervalo de confianca por bootstrap.

    Devolve a tabela de metricas, ordenada pela precisao media, e os scores de
    cada modelo, necessarios para os testes pareados.
    """
    linhas, scores = [], {}
    for nome, fabrica in zoo.items():
        resultado = fit_score(fabrica, X, y, split)
        scores[nome] = resultado["y_score"]
        metricas = classification_report(resultado["y_true"], resultado["y_score"])
        metricas["modelo"] = nome
        for metrica in ("roc_auc", "pr_auc"):
            ic = bootstrap_ci(resultado["y_true"], resultado["y_score"], metrica)
            metricas[f"{metrica}_ic_inf"] = ic["ic_inferior"]
            metricas[f"{metrica}_ic_sup"] = ic["ic_superior"]
        linhas.append(metricas)
        print(f"  {nome:<24} ROC-AUC {metricas['roc_auc']:.4f}  AP {metricas['pr_auc']:.4f}")

    tabela = pd.DataFrame(linhas).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return tabela, scores


def save_table(df: pd.DataFrame, nome: str) -> Path:
    destino = TABLES / f"{nome}.csv"
    df.to_csv(destino, index=False)
    return destino


def save_json(dados: dict, nome: str) -> Path:
    destino = RESULTS / f"{nome}.json"
    destino.write_text(json.dumps(dados, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return destino


def header(titulo: str) -> None:
    print(f"\n{'=' * 78}\n{titulo}\n{'=' * 78}")
