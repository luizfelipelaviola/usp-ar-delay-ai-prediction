"""Os tres comparadores e o pre-processamento.

Cada modelo e um ``Pipeline`` completo, do dado bruto a probabilidade, e por
isso a imputacao e a padronizacao sao ajustadas so com as faturas de treino.

Os comparadores vao do mais simples ao mais flexivel. Sao a regra que ordena
pela taxa de atraso historica do cliente, a regressao logistica e o XGBoost.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from arml import config


def build_preprocessor(numericas: list[str], categoricas: list[str], escalar: bool = True) -> ColumnTransformer:
    """Pre-processamento padrao.

    Os valores ausentes das colunas de historico indicam cliente sem fatura
    liquidada ate aquele momento. Sao imputados pela mediana, e a coluna
    ``cliente_sem_historico``, criada em ``features``, guarda a informacao de que
    o valor estava ausente.
    """
    etapas_num = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        etapas_num.append(("escalar", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(etapas_num), numericas),
            (
                "cat",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="most_frequent")),
                        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categoricas,
            ),
        ],
        remainder="drop",
    )


class HistoricalRateBaseline(BaseEstimator, ClassifierMixin):
    """Regra simples, que usa a taxa de atraso passada do proprio cliente.

    Reproduz o que um gestor faria sem modelo, olhando o historico do cliente no
    sistema contabil. Clientes sem historico recebem a taxa media de atraso do
    treino.
    """

    def __init__(self, coluna: str = "hist_taxa_atraso"):
        self.coluna = coluna

    def fit(self, X, y):
        self.prevalencia_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        taxa = np.asarray(X[self.coluna], dtype=float)
        p = np.where(np.isnan(taxa), self.prevalencia_, taxa)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.prevalencia_).astype(int)


def _xgboost(**kwargs):
    from xgboost import XGBClassifier

    parametros = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        random_state=config.SEED,
        n_jobs=-1,
    )
    parametros.update(kwargs)
    return XGBClassifier(**parametros)


#: A chave e o nome usado nas tabelas. O valor constroi um pipeline novo a cada
#: chamada, e cada experimento recebe uma instancia propria.
def model_zoo(numericas: list[str], categoricas: list[str]) -> dict[str, Callable[[], Pipeline]]:
    def com_pre(estimador, escalar: bool = False) -> Pipeline:
        return Pipeline(
            [
                ("pre", build_preprocessor(numericas, categoricas, escalar=escalar)),
                ("modelo", estimador),
            ]
        )

    return {
        "historico_cliente": lambda: Pipeline([("modelo", HistoricalRateBaseline())]),
        "regressao_logistica": lambda: com_pre(
            LogisticRegression(max_iter=2000, C=1.0, random_state=config.SEED), escalar=True
        ),
        "xgboost": lambda: com_pre(_xgboost()),
    }
