"""Emissao de faturas por dia da semana e outras medidas descritivas de um conjunto.

Num registro contabil, a emissao segue o calendario de trabalho, com menos
faturas no fim de semana e variacao entre os dias uteis. Uma distribuicao
uniforme nos sete dias, que o teste qui-quadrado nao rejeita, e compativel com
dado simulado.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

DIAS = ("segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo")


def assinaturas_de_emissao(df: pd.DataFrame) -> dict:
    """Mede, sobre o conjunto inteiro, as medidas citadas na Secao 5.1."""
    emissao = pd.to_datetime(df["InvoiceDate"]).dt.dayofweek
    contagem = emissao.value_counts().reindex(range(7), fill_value=0).sort_index()
    qui2, p = stats.chisquare(contagem.values)
    fim_de_semana = int(contagem.loc[5] + contagem.loc[6])
    por_cliente = df["customerID"].value_counts()
    return {
        "faturas_por_dia_da_semana": {DIAS[d]: int(contagem.loc[d]) for d in range(7)},
        "dia_de_pico": DIAS[int(contagem.idxmax())],
        "qui2_uniformidade": float(qui2),
        "p_uniformidade": float(p),
        "fracao_em_fim_de_semana": fim_de_semana / len(df),
        "valor_minimo": float(df["InvoiceAmount"].min()),
        "valor_maximo": float(df["InvoiceAmount"].max()),
        "assimetria_faturas_por_cliente": float(por_cliente.skew()),
        "faturas_do_maior_cliente": int(por_cliente.iloc[0]),
        "fracao_do_maior_cliente": float(por_cliente.iloc[0] / len(df)),
    }
