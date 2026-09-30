"""Carregamento do conjunto transacional.

Arquivo publico de 50.000 linhas, com campos no formato de um sistema de gestao
empresarial. Descartadas as linhas sem emissao ou vencimento e as de prazo fora
de 0 a 365 dias, restam 49.860 faturas de 1.460 clientes.

``clear_date`` e ``isOpen`` so existem depois do desfecho. Definem o rotulo e o
indicador de fatura liquidada, e a data de pagamento entra nos atributos apenas
quando anterior a emissao da fatura avaliada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arml.config import DATA_RAW

ARQUIVO = DATA_RAW / "transacional.csv"

#: Correspondencia entre as colunas do extrato e o esquema canonico do
#: pipeline, o mesmo que ``arml.data`` produz para o benchmark.
MAPEAMENTO = {
    "customerID": "cust_number",
    "invoiceNumber": "doc_id",
    "InvoiceDate": "posting_date",
    "DueDate": "due_in_date",
    "InvoiceAmount": "total_open_amount",
    "SettledDate": "clear_date",
    "countryCode": "business_code",
}
# O arquivo nao traz o pais do cliente; o codigo da unidade de negocio ocupa o
# lugar do atributo categorico ``pais``.

#: Moeda da fatura, usada na caracterizacao do conjunto.
EXTRAS = ["invoice_currency"]


def load_raw(path=ARQUIVO) -> pd.DataFrame:
    """Le o extrato bruto sem derivar nada."""
    df = pd.read_csv(path, low_memory=False)
    # O arquivo traz duas colunas com o mesmo nome; pandas as renomeia com
    # sufixo, e a segunda e redundante.
    return df.loc[:, ~df.columns.str.endswith(".1")]


def _para_data(serie: pd.Series) -> pd.Series:
    """Converte as varias representacoes de data do extrato.

    O arquivo mistura tres formatos, texto com hora para ``clear_date``, texto
    ISO para ``posting_date`` e inteiro no formato AAAAMMDD para ``due_in_date``.
    Tentar um formato unico produziria valores ausentes silenciosos.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_datetime(serie.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(serie, errors="coerce")


def load_dataset(path=ARQUIVO) -> pd.DataFrame:
    """Carrega no esquema canonico, com alvos derivados.

    Colunas de saida, alem das do esquema canonico:

        ``evento``  1 quando a liquidacao foi registrada, 0 quando a fatura consta em aberto
    """
    bruto = load_raw(path)
    out = pd.DataFrame(index=bruto.index)

    for canonico, origem in MAPEAMENTO.items():
        out[canonico] = bruto[origem]
    for extra in EXTRAS:
        out[extra] = bruto[extra]

    for coluna in ("InvoiceDate", "DueDate", "SettledDate"):
        out[coluna] = _para_data(out[coluna])

    out["evento"] = (bruto["isOpen"] == 0).astype(int)
    out["customerID"] = out["customerID"].astype(str)
    out["InvoiceAmount"] = pd.to_numeric(out["InvoiceAmount"], errors="coerce")

    # Descarta o que nao tem data de emissao ou de vencimento, porque sem elas
    # nao ha como definir prazo nem alinhar o corte temporal.
    out = out.dropna(subset=["InvoiceDate", "DueDate", "InvoiceAmount"]).copy()
    out["prazo_dias"] = (out["DueDate"] - out["InvoiceDate"]).dt.days
    # Prazo negativo ou acima de um ano indica lancamento retroativo ou erro de digitacao.
    out = out[(out["prazo_dias"] >= 0) & (out["prazo_dias"] <= 365)].copy()

    liquidada = out["evento"] == 1
    out.loc[liquidada & out["SettledDate"].isna(), "evento"] = 0

    # A data de observacao do conjunto e a do ultimo evento registrado.
    observacao = pd.concat([out["SettledDate"].dropna(), out["InvoiceDate"]]).max()

    liquidada = out["evento"] == 1
    out["y_settle"] = np.where(liquidada, (out["SettledDate"] - out["InvoiceDate"]).dt.days, np.nan)

    atraso = np.where(liquidada, (out["SettledDate"] - out["DueDate"]).dt.days, np.nan)
    out["y_days"] = np.clip(atraso, 0, None)
    out["y_late"] = np.where(liquidada, (out["y_days"] > 0).astype(float), np.nan)

    # Colunas que o pipeline espera e que este arquivo nao traz. Recebem um valor
    # constante para nao interferir na imputacao.
    out["PaperlessBill"] = "Electronic"
    out["PaperlessDate"] = out["InvoiceDate"]
    out["Disputed"] = "No"
    out["countryCode"] = out["countryCode"].astype(str)

    out = out.sort_values(["InvoiceDate", "invoiceNumber"], kind="mergesort").reset_index(drop=True)
    out.attrs["data_observacao"] = observacao

    _validar(out)
    return out


def _validar(df: pd.DataFrame) -> None:
    """Falha cedo diante de inconsistencia que invalidaria os experimentos."""
    if (df["DueDate"] < df["InvoiceDate"]).any():
        raise ValueError("existe fatura com vencimento anterior a emissao")
    liquidadas = df[df["evento"] == 1]
    if (liquidadas["SettledDate"] < liquidadas["InvoiceDate"]).any():
        raise ValueError("existe fatura liquidada antes de emitida")
    if not df["InvoiceDate"].is_monotonic_increasing:
        raise ValueError("o conjunto nao esta ordenado por data de emissao")
    if df.loc[df["evento"] == 1, "y_late"].isna().any():
        raise ValueError("fatura liquidada sem rotulo de atraso")


def describe(df: pd.DataFrame) -> dict:
    """Resumo descritivo do conjunto transacional."""
    liquidadas = df[df["evento"] == 1]
    atrasadas = liquidadas.loc[liquidadas["y_late"] == 1, "y_days"]
    return {
        "n_faturas": len(df),
        "n_liquidadas": int((df["evento"] == 1).sum()),
        "n_em_aberto": int((df["evento"] == 0).sum()),
        "fracao_em_aberto": float((df["evento"] == 0).mean()),
        "n_clientes": int(df["customerID"].nunique()),
        "inicio": str(df["InvoiceDate"].min().date()),
        "fim": str(df["InvoiceDate"].max().date()),
        "data_observacao": str(pd.Timestamp(df.attrs.get("data_observacao")).date()),
        "prazos_distintos": int(df["prazo_dias"].nunique()),
        "prazo_mediano": int(df["prazo_dias"].median()),
        "taxa_atraso": float(liquidadas["y_late"].mean()),
        "atraso_mediano_atrasadas": float(atrasadas.median()) if len(atrasadas) else 0.0,
        "atraso_maximo": int(liquidadas["y_days"].max()),
        "valor_mediano": float(df["InvoiceAmount"].median()),
    }
