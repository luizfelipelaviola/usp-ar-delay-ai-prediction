"""Carregamento do conjunto candidato excluido na selecao de dados.

O conjunto atende ao criterio minimo de campos, mas o rotulo de atraso depende
mecanicamente do dia da semana do vencimento. Por isso ele nao integra a
avaliacao experimental e e usado apenas para documentar esse efeito de
calendario.

``Clearing_date`` so existe depois do desfecho e entra apenas no rotulo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arml.config import DATA_RAW

ARQUIVO = DATA_RAW / "kaggle_ar_delay.csv"

#: Correspondencia com o esquema canonico do pipeline.
MAPEAMENTO = {
    "customerID": "Cust_Num",
    "invoiceNumber": "Document_No",
    "InvoiceDate": "Doc_Date",
    "DueDate": "Net_Due_Date",
    "InvoiceAmount": "Amount",
    "SettledDate": "Clearing_date",
}


def load_raw(path=ARQUIVO) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} ausente. Obtenha os dados com: python scripts/baixar_dados.py")
    return pd.read_csv(path, low_memory=False)


def load_dataset(path=ARQUIVO) -> pd.DataFrame:
    """Carrega no esquema canonico, com alvos derivados. Toda linha tem liquidacao registrada."""
    bruto = load_raw(path)
    out = pd.DataFrame(index=bruto.index)

    for canonico, origem in MAPEAMENTO.items():
        out[canonico] = bruto[origem]

    for coluna in ("InvoiceDate", "DueDate", "SettledDate"):
        out[coluna] = pd.to_datetime(out[coluna], format="mixed", errors="coerce")

    out["customerID"] = out["customerID"].astype(str)
    out["InvoiceAmount"] = pd.to_numeric(out["InvoiceAmount"], errors="coerce")

    out = out.dropna(subset=["InvoiceDate", "DueDate", "SettledDate", "InvoiceAmount"]).copy()
    out["prazo_dias"] = (out["DueDate"] - out["InvoiceDate"]).dt.days
    # Prazo negativo indica lancamento retroativo; acima de um ano, erro de
    # digitacao. Ambos sao descartados, como no conjunto transacional.
    out = out[(out["prazo_dias"] >= 0) & (out["prazo_dias"] <= 365)].copy()

    atraso = (out["SettledDate"] - out["DueDate"]).dt.days
    out["y_days"] = np.clip(atraso, 0, None).astype(float)
    out["y_late"] = (out["y_days"] > 0).astype(float)

    return out.sort_values(["InvoiceDate", "invoiceNumber"], kind="mergesort").reset_index(drop=True)


def describe(df: pd.DataFrame) -> dict:
    atrasadas = df.loc[df["y_late"] == 1, "y_days"]
    return {
        "n_faturas": int(len(df)),
        "n_clientes": int(df["customerID"].nunique()),
        "inicio": str(df["InvoiceDate"].min().date()),
        "fim": str(df["InvoiceDate"].max().date()),
        "prazos_distintos": int(df["prazo_dias"].nunique()),
        "prazo_mediano": float(df["prazo_dias"].median()),
        "taxa_atraso": float(df["y_late"].mean()),
        "atraso_mediano_atrasadas": float(atrasadas.median()) if len(atrasadas) else 0.0,
        "atraso_maximo": int(df["y_days"].max()),
        "valor_mediano": float(df["InvoiceAmount"].median()),
    }
