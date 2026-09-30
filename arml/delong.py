"""Teste de DeLong para a diferenca entre duas ROC-AUC calculadas sobre as mesmas faturas."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    """Posto medio, com tratamento correto de empates.

    Escores discretos ou saturados geram muitos empates, e o posto simples
    enviesaria a variancia.
    """
    ordem = np.argsort(x)
    ordenado = x[ordem]
    n = len(x)
    postos = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n - 1 and ordenado[j + 1] == ordenado[i]:
            j += 1
        postos[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1

    saida = np.empty(n, dtype=float)
    saida[ordem] = postos
    return saida


def delong_variance(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estima a AUC e a matriz de covariancia pelo algoritmo rapido de DeLong.

    :param y_true: rotulos binarios
    :param scores: matriz (n_modelos, n_amostras) com os escores de cada modelo
    :returns: (aucs, matriz de covariancia entre as AUCs)
    """
    y_true = np.asarray(y_true)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))

    positivos = scores[:, y_true == 1]
    negativos = scores[:, y_true == 0]
    m, n = positivos.shape[1], negativos.shape[1]
    k = scores.shape[0]

    # Componentes de estrutura de DeLong. Os postos separados e combinados
    # permitem escrever a AUC como media de contribuicoes por observacao.
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(positivos[r])
        ty[r] = _midrank(negativos[r])
        tz[r] = _midrank(np.concatenate([positivos[r], negativos[r]]))

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    covariancia = np.cov(v01) / m + np.cov(v10) / n
    return aucs, covariancia


def delong_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """Teste de DeLong para a diferenca entre duas AUCs correlacionadas.

    Devolve a diferenca, seu erro padrao, o intervalo de confianca de 95% e o
    valor-p bilateral.
    """
    aucs, cov = delong_variance(y_true, np.vstack([score_a, score_b]))
    diferenca = float(aucs[0] - aucs[1])
    # Variancia da diferenca de duas estimativas correlacionadas.
    variancia = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    erro_padrao = float(np.sqrt(max(variancia, 1e-300)))

    z = diferenca / erro_padrao
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diferenca": diferenca,
        "erro_padrao": erro_padrao,
        "z": float(z),
        "p_valor": p,
        "ic_inferior": diferenca - 1.96 * erro_padrao,
        "ic_superior": diferenca + 1.96 * erro_padrao,
        "significativa_5pct": bool(p < 0.05),
    }
