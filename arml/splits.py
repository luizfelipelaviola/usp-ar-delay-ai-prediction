"""Holdout temporal e restricao do treino aos rotulos ja conhecidos no corte.

O modelo e treinado com as faturas mais antigas e avaliado com as mais recentes,
como no uso real. Uma divisao aleatoria colocaria faturas futuras no treino.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arml.config import TEST_FRACTION


@dataclass(frozen=True)
class Split:
    """Uma particao nomeada, com indices posicionais de treino e teste."""

    nome: str
    treino: np.ndarray
    teste: np.ndarray
    descricao: str


def out_of_time(df: pd.DataFrame, fracao_teste: float = TEST_FRACTION) -> Split:
    """Divide a linha do tempo num ponto unico, com o passado no treino e o futuro no teste.

    O corte cai entre duas datas de emissao, e as faturas de uma mesma data ficam
    todas no treino ou todas no teste. Cortar por posicao dividiria essas faturas
    e deixaria passar informacao do teste para o treino pelos atributos de
    faturas em aberto.
    """
    datas = df["InvoiceDate"].sort_values().unique()
    corte = pd.Timestamp(datas[int(len(datas) * (1 - fracao_teste))])

    treino = np.flatnonzero(df["InvoiceDate"].to_numpy() < np.datetime64(corte))
    teste = np.flatnonzero(df["InvoiceDate"].to_numpy() >= np.datetime64(corte))
    return Split(
        nome="holdout_temporal",
        treino=treino,
        teste=teste,
        descricao=f"treino ate {corte.date()}, teste a partir de {corte.date()}",
    )


def maturar(df: pd.DataFrame, split: Split) -> Split:
    """Retira do treino as faturas cujo desfecho ainda nao era conhecido no corte.

    O corte por data de emissao garante que os atributos usam so o passado, mas
    nao garante que o rotulo do treino ja fosse conhecido no corte. Uma fatura
    emitida antes dele, com vencimento e pagamento depois, so revela no futuro se
    atrasou. Quando o vencimento ja passou, o atraso e conhecido mesmo sem
    pagamento, e a fatura fica no treino.
    """
    corte = df["InvoiceDate"].iloc[split.teste].min()
    treino = df.iloc[split.treino]
    desconhecido = (~(treino["SettledDate"] < corte) & (treino["DueDate"] >= corte)).to_numpy()
    return Split(
        nome=f"{split.nome}_maduro",
        treino=split.treino[~desconhecido],
        teste=split.teste,
        descricao=f"{split.descricao}, treino apenas com desfecho conhecido no corte",
    )
