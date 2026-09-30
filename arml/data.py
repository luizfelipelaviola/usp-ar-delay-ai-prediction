"""Carregamento e validacao do benchmark publico no esquema usado pelo pipeline.

A data de pagamento (SettledDate) define o rotulo. Nos atributos, entra apenas a
das faturas ja pagas antes da emissao da fatura avaliada.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from arml.config import AR_CSV, LATE_THRESHOLD_DAYS

DATE_COLUMNS = ["PaperlessDate", "InvoiceDate", "DueDate", "SettledDate"]


@dataclass(frozen=True)
class DatasetInfo:
    """Resumo descritivo do conjunto carregado, usado nas tabelas do trabalho."""

    n_faturas: int
    n_clientes: int
    inicio: pd.Timestamp
    fim: pd.Timestamp
    taxa_atraso: float
    prazo_dias: int
    valor_mediano: float

    def as_dict(self) -> dict:
        return {
            "n_faturas": self.n_faturas,
            "n_clientes": self.n_clientes,
            "inicio": str(self.inicio.date()),
            "fim": str(self.fim.date()),
            "taxa_atraso": self.taxa_atraso,
            "prazo_dias": self.prazo_dias,
            "valor_mediano": self.valor_mediano,
        }


def load_raw(path=AR_CSV) -> pd.DataFrame:
    """Le o CSV bruto e converte as colunas de data, sem derivar nada."""
    df = pd.read_csv(path)
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y", errors="raise")
    return df


def build_targets(df: pd.DataFrame, threshold: int = LATE_THRESHOLD_DAYS) -> pd.DataFrame:
    """Deriva o rotulo e as grandezas de atraso usadas nos atributos e na caracterizacao.

    ``y_late``    binaria, o rotulo do trabalho: liquidacao apos o vencimento
    ``y_days``    contagem de dias de atraso, zero quando em dia
    ``y_settle``  dias entre emissao e liquidacao, usado no historico do cliente
    """
    out = df.copy()

    dias_atraso = (out["SettledDate"] - out["DueDate"]).dt.days
    # Pagamento antecipado conta como pagamento em dia.
    out["y_days"] = dias_atraso.clip(lower=0)
    out["y_late"] = (out["y_days"] > threshold).astype(int)
    out["y_settle"] = (out["SettledDate"] - out["InvoiceDate"]).dt.days
    out["prazo_dias"] = (out["DueDate"] - out["InvoiceDate"]).dt.days

    return out


def load_dataset(path=AR_CSV) -> pd.DataFrame:
    """Carrega o conjunto pronto para o pipeline, ordenado no tempo.

    Os atributos de historico e o holdout temporal dependem da ordenacao por
    data de emissao.
    """
    df = build_targets(load_raw(path))
    df = df.sort_values(["InvoiceDate", "invoiceNumber"], kind="mergesort").reset_index(drop=True)
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    """Falha cedo diante de inconsistencia que invalidaria os experimentos."""
    if df.isna().any().any():
        colunas = df.columns[df.isna().any()].tolist()
        raise ValueError(f"valores ausentes nas colunas: {colunas}")
    if (df["DueDate"] < df["InvoiceDate"]).any():
        raise ValueError("existe fatura com vencimento anterior a emissao")
    if (df["SettledDate"] < df["InvoiceDate"]).any():
        raise ValueError("existe fatura liquidada antes de emitida")
    if not df["InvoiceDate"].is_monotonic_increasing:
        raise ValueError("o conjunto nao esta ordenado por data de emissao")


def describe(df: pd.DataFrame) -> DatasetInfo:
    """Resumo descritivo usado na secao de caracterizacao da amostra."""
    return DatasetInfo(
        n_faturas=len(df),
        n_clientes=df["customerID"].nunique(),
        inicio=df["InvoiceDate"].min(),
        fim=df["InvoiceDate"].max(),
        taxa_atraso=float(df["y_late"].mean()),
        prazo_dias=int(df["prazo_dias"].mode().iloc[0]),
        valor_mediano=float(df["InvoiceAmount"].median()),
    )
