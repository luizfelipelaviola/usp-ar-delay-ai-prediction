"""Atributos de cada fatura, restritos ao que era conhecido na data de emissao.

Para pontuar uma fatura emitida em t, o modelo so pode usar o que a empresa sabia
em t. Esse criterio e mais restritivo do que usar todas as faturas anteriores,
como mostra o exemplo.

    fatura A  emitida em 10/jan, paga em 25/mar
    fatura B  emitida em 05/fev

Em 05/fev a fatura A ja existe, mas ainda esta em aberto, e seu atraso e
desconhecido. Contar A no historico de atraso do cliente ao pontuar B usaria
informacao do futuro, e o efeito e maior nos clientes que demoram a pagar.

As faturas anteriores contribuem de dois modos.

    ja liquidadas em t   desfecho (atraso e tempo ate o pagamento)
    em aberto em t       apenas existencia (quantidade, valor e se ja venceu)

O segundo grupo corresponde ao campo Balance do QuickBooks e ao relatorio de
aging, informacao disponivel no sistema contabil.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Meia-vida, em numero de faturas, do peso da media exponencial do historico.
#: Valor baixo privilegia o comportamento recente do cliente.
EWMA_HALFLIFE = 3.0


def build_features(df: pd.DataFrame, include_disputed: bool = False) -> pd.DataFrame:
    """Monta a matriz de atributos respeitando o conjunto de informacao em t.

    :param df: saida de ``arml.data.load_dataset``, ordenada por data de emissao
    :param include_disputed: inclui a coluna ``Disputed``, cuja data de origem e
        desconhecida e por isso pode conter informacao posterior ao desfecho
    """
    out = df.copy()

    out = _calendar_features(out)
    out = _invoice_features(out)
    out = _customer_history(out)
    out = _portfolio_state(out)

    if include_disputed:
        out["disputed"] = (out["Disputed"] == "Yes").astype(int)

    return out


# ---------------------------------------------------------------------------
# Atributos deterministicos da propria fatura
# ---------------------------------------------------------------------------


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sazonalidade. Conhecidos por construcao no momento da emissao."""
    out = df.copy()
    emissao, vencimento = out["InvoiceDate"], out["DueDate"]

    out["mes_emissao"] = emissao.dt.month
    out["trimestre_emissao"] = emissao.dt.quarter
    out["dia_semana_emissao"] = emissao.dt.dayofweek
    out["dia_mes_emissao"] = emissao.dt.day
    out["dia_semana_vencimento"] = vencimento.dt.dayofweek
    out["mes_vencimento"] = vencimento.dt.month

    # Vencimento em fim de semana leva o pagamento ao dia util seguinte e gera
    # atraso de um ou dois dias sem relacao com risco de credito.
    out["vencimento_fim_semana"] = (vencimento.dt.dayofweek >= 5).astype(int)
    out["vencimento_fim_mes"] = vencimento.dt.is_month_end.astype(int)
    out["vencimento_fim_trimestre"] = vencimento.dt.is_quarter_end.astype(int)
    return out


def _invoice_features(df: pd.DataFrame) -> pd.DataFrame:
    """Atributos do documento em si."""
    out = df.copy()
    out["valor"] = out["InvoiceAmount"]
    # O valor e assimetrico a direita; o logaritmo estabiliza a escala para os
    # modelos lineares. Os modelos de arvore sao invariantes a essa transformacao.
    out["log_valor"] = np.log1p(out["InvoiceAmount"])
    # No benchmark, a adesao ao faturamento eletronico ocorre depois da emissao
    # em cerca de metade das faturas, e `PaperlessBill` registra so a situacao
    # final. Em t sabe-se apenas se o cliente ja havia aderido. Quando ainda nao
    # havia, o indicador e o tempo de adesao valem zero.
    dias_adesao = (out["InvoiceDate"] - out["PaperlessDate"]).dt.days
    ja_aderiu = dias_adesao >= 0
    out["adesao_eletronica_ativa"] = ja_aderiu.astype(int)
    out["dias_desde_adesao_eletronica"] = dias_adesao.where(ja_aderiu, 0)
    out["fatura_eletronica"] = (
        (out["PaperlessBill"] == "Electronic") & ja_aderiu).astype(int)

    out["pais"] = out["countryCode"].astype(str)
    return out


# ---------------------------------------------------------------------------
# Historico do cliente
# ---------------------------------------------------------------------------


def _customer_history(df: pd.DataFrame) -> pd.DataFrame:
    """Historico de pagamento do cliente, restrito ao que ja era observavel.

    Para cada fatura i do cliente c emitida em t, agrega somente as faturas de c
    com ``SettledDate < t``. A comparacao e estrita, porque uma fatura paga no
    proprio dia t so aparece no sistema depois do lancamento.
    """
    out = df.copy()
    colunas = [
        "hist_n",
        "hist_taxa_atraso",
        "hist_media_dias_atraso",
        "hist_max_dias_atraso",
        "hist_desvio_dias_atraso",
        "hist_ewma_dias_atraso",
        "hist_media_dias_liquidacao",
        "hist_media_valor",
        "hist_dias_desde_ultima_liquidacao",
    ]
    for col in colunas:
        out[col] = np.nan

    for _, indices in out.groupby("customerID", sort=False).groups.items():
        idx = np.asarray(indices)
        emissao = out.loc[idx, "InvoiceDate"].to_numpy()
        liquidacao = out.loc[idx, "SettledDate"].to_numpy()
        atraso = out.loc[idx, "y_days"].to_numpy(dtype=float)
        tempo_liq = out.loc[idx, "y_settle"].to_numpy(dtype=float)
        valor = out.loc[idx, "InvoiceAmount"].to_numpy(dtype=float)

        for pos, linha in enumerate(idx):
            t = emissao[pos]
            conhecidas = liquidacao < t
            n = int(conhecidas.sum())
            out.at[linha, "hist_n"] = n
            if n == 0:
                continue

            atrasos = atraso[conhecidas]
            out.at[linha, "hist_taxa_atraso"] = float((atrasos > 0).mean())
            out.at[linha, "hist_media_dias_atraso"] = float(atrasos.mean())
            out.at[linha, "hist_max_dias_atraso"] = float(atrasos.max())
            out.at[linha, "hist_desvio_dias_atraso"] = float(atrasos.std(ddof=0))
            out.at[linha, "hist_media_dias_liquidacao"] = float(tempo_liq[conhecidas].mean())
            out.at[linha, "hist_media_valor"] = float(valor[conhecidas].mean())

            # Media exponencial na ordem de pagamento, em que o comportamento
            # recente pesa mais que o antigo.
            ordem = np.argsort(liquidacao[conhecidas])
            serie = pd.Series(atrasos[ordem])
            out.at[linha, "hist_ewma_dias_atraso"] = float(serie.ewm(halflife=EWMA_HALFLIFE).mean().iloc[-1])

            ultima = liquidacao[conhecidas].max()
            out.at[linha, "hist_dias_desde_ultima_liquidacao"] = float((t - ultima) / np.timedelta64(1, "D"))

    # Tempo de relacionamento e frequencia usam apenas datas de emissao, que sao
    # conhecidas imediatamente e portanto nao exigem o corte por liquidacao.
    grupo = out.groupby("customerID", sort=False)["InvoiceDate"]
    out["cliente_antiguidade_dias"] = (out["InvoiceDate"] - grupo.transform("min")).dt.days
    out["dias_desde_fatura_anterior"] = grupo.diff().dt.days
    out["indice_fatura_cliente"] = out.groupby("customerID", sort=False).cumcount()

    # Cliente sem fatura liquidada e o caso de partida a frio. O sinalizador separa "sem historico" de "historico com valor zero".
    out["cliente_sem_historico"] = (out["hist_n"].fillna(0) == 0).astype(int)
    return out


def _portfolio_state(df: pd.DataFrame) -> pd.DataFrame:
    """Faturas do cliente em aberto no momento da emissao.

    Conta as faturas ja emitidas e ainda nao liquidadas em t, como o campo Balance
    do QuickBooks e o relatorio de aging. Usa apenas quantidade, valor e
    vencimento, nunca o desfecho.
    """
    out = df.copy()
    out["aberto_qtd"] = 0
    out["aberto_valor"] = 0.0
    out["aberto_vencido_qtd"] = 0

    for _, indices in out.groupby("customerID", sort=False).groups.items():
        idx = np.asarray(indices)
        emissao = out.loc[idx, "InvoiceDate"].to_numpy()
        vencimento = out.loc[idx, "DueDate"].to_numpy()
        liquidacao = out.loc[idx, "SettledDate"].to_numpy()
        valor = out.loc[idx, "InvoiceAmount"].to_numpy(dtype=float)

        for pos, linha in enumerate(idx):
            t = emissao[pos]
            em_aberto = (emissao < t) & ~(liquidacao < t)
            out.at[linha, "aberto_qtd"] = int(em_aberto.sum())
            out.at[linha, "aberto_valor"] = float(valor[em_aberto].sum())
            # Faturas vencidas e ainda em aberto, o sinal mais direto de dificuldade
            # de caixa que a empresa ve no proprio sistema.
            out.at[linha, "aberto_vencido_qtd"] = int((em_aberto & (vencimento < t)).sum())

    return out


# ---------------------------------------------------------------------------
# Selecao de colunas
# ---------------------------------------------------------------------------

NUMERICAS = [
    "log_valor",
    "valor",
    "prazo_dias",
    "mes_emissao",
    "trimestre_emissao",
    "dia_semana_emissao",
    "dia_mes_emissao",
    "dia_semana_vencimento",
    "mes_vencimento",
    "vencimento_fim_semana",
    "vencimento_fim_mes",
    "vencimento_fim_trimestre",
    "fatura_eletronica",
    "dias_desde_adesao_eletronica",
    "adesao_eletronica_ativa",
    "hist_n",
    "hist_taxa_atraso",
    "hist_media_dias_atraso",
    "hist_max_dias_atraso",
    "hist_desvio_dias_atraso",
    "hist_ewma_dias_atraso",
    "hist_media_dias_liquidacao",
    "hist_media_valor",
    "hist_dias_desde_ultima_liquidacao",
    "cliente_antiguidade_dias",
    "dias_desde_fatura_anterior",
    "indice_fatura_cliente",
    "cliente_sem_historico",
    "aberto_qtd",
    "aberto_valor",
    "aberto_vencido_qtd",
]

CATEGORICAS = ["pais"]


# ---------------------------------------------------------------------------
# Variantes usadas para medir vazamento
# ---------------------------------------------------------------------------


def build_features_leaky(df: pd.DataFrame) -> pd.DataFrame:
    """Controle positivo de vazamento, com colunas posteriores ao desfecho.

    Serve apenas para mostrar que as metricas sobem quando o modelo recebe
    informacao do futuro, em contraste com os resultados obtidos sob o protocolo.
    """
    out = build_features(df, include_disputed=True)
    out["dias_ate_liquidacao"] = out["y_settle"]
    return out


NUMERICAS_LEAKY = NUMERICAS + ["disputed", "dias_ate_liquidacao"]
