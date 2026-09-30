"""Metricas de discriminacao e testes de comparacao entre modelos.

A metrica principal e a precisao media (AP), calculada por
``average_precision_score``. Ela resume a curva de precisao e revocacao e avalia
a ordenacao das faturas usada na cobranca. A ROC-AUC e complementar.

Cada metrica vem com intervalo de confianca por bootstrap estratificado, e cada
par de modelos e comparado por bootstrap pareado, sobre as mesmas faturas de
teste. A correcao de Holm controla o erro nas comparacoes multiplas.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from arml.config import N_BOOTSTRAP, SEED

FUNCOES = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}


def classification_report(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Precisao media e ROC-AUC de um score de risco."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "prevalencia": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrica: str = "roc_auc",
    n_reamostras: int = N_BOOTSTRAP,
    alfa: float = 0.05,
) -> dict[str, float]:
    """Intervalo de confianca percentil por bootstrap sobre o conjunto de teste.

    Reamostragem estratificada por classe, para que toda reamostra preserve a
    prevalencia e as metricas continuem definidas.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(SEED)
    idx_pos = np.flatnonzero(y_true == 1)
    idx_neg = np.flatnonzero(y_true == 0)

    fn = FUNCOES[metrica]

    amostras = np.empty(n_reamostras)
    for i in range(n_reamostras):
        sel = np.concatenate(
            [
                rng.choice(idx_pos, size=len(idx_pos), replace=True),
                rng.choice(idx_neg, size=len(idx_neg), replace=True),
            ]
        )
        amostras[i] = fn(y_true[sel], y_score[sel])

    return {
        "metrica": metrica,
        "estimativa": float(fn(y_true, y_score)),
        "ic_inferior": float(np.percentile(amostras, 100 * alfa / 2)),
        "ic_superior": float(np.percentile(amostras, 100 * (1 - alfa / 2))),
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metrica: str = "roc_auc",
    n_reamostras: int = N_BOOTSTRAP,
) -> dict[str, float]:
    """Compara dois modelos no mesmo conjunto de teste, por bootstrap pareado.

    As duas predicoes vem das mesmas faturas, e seus erros sao correlacionados.
    Reamostrar cada modelo de forma independente ignoraria essa correlacao e daria
    um intervalo largo demais para a diferenca.

    Devolve a diferenca A menos B com intervalo, e o valor-p bilateral empirico.
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(SEED)
    fn = FUNCOES[metrica]

    observada = fn(y_true, score_a) - fn(y_true, score_b)
    idx_pos = np.flatnonzero(y_true == 1)
    idx_neg = np.flatnonzero(y_true == 0)

    diferencas = np.empty(n_reamostras)
    for i in range(n_reamostras):
        sel = np.concatenate(
            [
                rng.choice(idx_pos, size=len(idx_pos), replace=True),
                rng.choice(idx_neg, size=len(idx_neg), replace=True),
            ]
        )
        diferencas[i] = fn(y_true[sel], score_a[sel]) - fn(y_true[sel], score_b[sel])

    # Valor-p bilateral, dado pela proporcao de reamostras cuja diferenca troca de sinal
    # em relacao a diferenca observada, duplicada para cobrir as duas caudas.
    p = 2 * min(float(np.mean(diferencas <= 0)), float(np.mean(diferencas >= 0)))
    return {
        "metrica": metrica,
        "diferenca": float(observada),
        "ic_inferior": float(np.percentile(diferencas, 2.5)),
        "ic_superior": float(np.percentile(diferencas, 97.5)),
        "p_valor": min(1.0, p),
        "significativo_5pct": bool(min(1.0, p) < 0.05),
    }


def holm_correction(p_valores, alfa: float = 0.05) -> list[dict]:
    """Correcao de Holm-Bonferroni para multiplas comparacoes.

    Controla a taxa de erro da familia de comparacoes com mais poder que o
    Bonferroni simples. Os valores-p sao ordenados, o k-esimo e comparado com alfa
    dividido por (m - k + 1), e o procedimento para na primeira comparacao nao
    rejeitada.
    """
    p_valores = list(p_valores)
    m = len(p_valores)
    ordem = np.argsort(p_valores)
    resultado: list[dict] = [{} for _ in range(m)]
    rejeitando = True

    for k, i in enumerate(ordem):
        limiar = alfa / (m - k)
        rejeita = rejeitando and p_valores[i] < limiar
        if not rejeita:
            rejeitando = False
        resultado[i] = {
            "p_original": float(p_valores[i]),
            "posicao": int(k + 1),
            "limiar_holm": float(limiar),
            "significativo_apos_holm": bool(rejeita),
        }
    return resultado
