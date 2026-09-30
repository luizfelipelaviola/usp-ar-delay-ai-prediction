"""Caracterizacao dos conjuntos de dados.

Produz os numeros descritivos do trabalho para os dois conjuntos avaliados e para
o conjunto candidato excluido na selecao de dados.

    benchmark            tamanho, prazo, taxa e distribuicao do atraso, emissao
                         por dia da semana e as duas premissas dos atributos
    transacional         o mesmo, mais moedas, arquivo bruto e o efeito de
                         calendario do vencimento
    conjunto_excluido    o efeito de calendario que motivou a exclusao

As duas premissas medidas sao a dependencia do atraso em relacao ao cliente
(fracao da variancia da taxa de atraso atribuivel ao cliente) e o valor do
historico para prever o desfecho (correlacao de Spearman entre a taxa de atraso
historica, calculada so com o passado, e o desfecho da fatura).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from arml import data, data_transacional, data_kaggle_ar
from arml.caracterizacao import DIAS, assinaturas_de_emissao
from arml.evaluate import header, save_json
from arml.features import build_features

FAIXAS = ["em_dia", "1_30", "31_60", "61_90", "90_mais"]


def faixas_de_atraso(liquidadas: pd.DataFrame) -> dict:
    faixas = pd.cut(liquidadas["y_days"], [-0.1, 0, 30, 60, 90, 1e6], labels=FAIXAS)
    contagem = faixas.value_counts().reindex(FAIXAS, fill_value=0)
    return {nome: {"faturas": int(n), "fracao": n / len(liquidadas)} for nome, n in contagem.items()}


def premissas(liquidadas: pd.DataFrame, minimo_faturas: int) -> dict:
    por_cliente = liquidadas.groupby("customerID")["y_late"].agg(n_atrasos="sum", n_faturas="count")
    por_cliente = por_cliente[por_cliente["n_faturas"] >= minimo_faturas]
    taxa = por_cliente["n_atrasos"] / por_cliente["n_faturas"]
    tabela = np.column_stack([por_cliente["n_atrasos"], por_cliente["n_faturas"] - por_cliente["n_atrasos"]])
    qui2, p_qui2, gl, _ = stats.chi2_contingency(tabela)

    media = float(liquidadas["y_late"].mean())
    var_entre = float(taxa.var(ddof=1))
    var_dentro = media * (1 - media) / float(por_cliente["n_faturas"].mean())
    icc = (var_entre - var_dentro) / var_entre

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = build_features(liquidadas)
    com_historico = feat[feat["hist_n"] > 0]
    rho, p_rho = stats.spearmanr(com_historico["hist_taxa_atraso"], com_historico["y_late"])
    return {
        "qui2_homogeneidade": float(qui2),
        "gl": int(gl),
        "p_homogeneidade": float(p_qui2),
        "variancia_explicada_pelo_cliente": float(icc),
        "spearman_historico": float(rho),
        "p_spearman": float(p_rho),
    }


def calendario_do_vencimento(liquidadas: pd.DataFrame) -> dict:
    dia = liquidadas["DueDate"].dt.dayofweek
    atrasou = liquidadas["y_late"].astype(bool)
    por_dia = pd.DataFrame({"dia": dia, "atrasou": atrasou}).groupby("dia")["atrasou"].agg(["size", "mean"])
    fim_de_semana = dia >= 5
    atrasadas_fds = liquidadas[fim_de_semana & atrasou]
    return {
        "por_dia_do_vencimento": {
            DIAS[d]: {"faturas": int(por_dia.loc[d, "size"]), "taxa_atraso": float(por_dia.loc[d, "mean"])}
            for d in por_dia.index
        },
        "taxa_fim_de_semana": float(atrasou[fim_de_semana].mean()),
        "taxa_dia_util": float(atrasou[~fim_de_semana].mean()),
        "fracao_atraso_ate_dois_dias_no_fim_de_semana": float((atrasadas_fds["y_days"] <= 2).mean()),
    }


def benchmark() -> dict:
    df = data.load_dataset()
    atrasadas = df.loc[df["y_late"] == 1, "y_days"]
    return {
        "amostra": data.describe(df).as_dict(),
        "prazos_distintos": int(df["prazo_dias"].nunique()),
        "atraso_mediano_atrasadas": float(atrasadas.median()),
        "atraso_maximo": int(df["y_days"].max()),
        "faixas_de_atraso": faixas_de_atraso(df),
        "assinaturas_de_emissao": assinaturas_de_emissao(df),
        "premissas": premissas(df, minimo_faturas=1),
    }


def transacional() -> dict:
    bruto = data_transacional.load_raw()
    completo = data_transacional.load_dataset()
    liquidadas = completo[completo["evento"] == 1].reset_index(drop=True)
    return {
        "arquivo_bruto": {
            "linhas": int(len(bruto)),
            "clientes": int(bruto["cust_number"].nunique()),
            "linhas_descartadas": int(len(bruto) - len(completo)),
        },
        "amostra": data_transacional.describe(completo),
        "faturas_por_moeda": {m: int(n) for m, n in completo["invoice_currency"].value_counts().items()},
        "faixas_de_atraso": faixas_de_atraso(liquidadas),
        "assinaturas_de_emissao": assinaturas_de_emissao(completo),
        "premissas": premissas(liquidadas, minimo_faturas=5),
        "calendario": calendario_do_vencimento(liquidadas),
    }


def conjunto_excluido() -> dict:
    bruto = data_kaggle_ar.load_raw()
    df = data_kaggle_ar.load_dataset()
    return {
        "arquivo_bruto": {
            "linhas": int(len(bruto)),
            "clientes": int(bruto["Cust_Num"].nunique()),
        },
        "amostra": data_kaggle_ar.describe(df),
        "assinaturas_de_emissao": assinaturas_de_emissao(df),
        "calendario": calendario_do_vencimento(df),
    }


def main() -> None:
    header("CARACTERIZACAO DOS CONJUNTOS DE DADOS")
    resultado = {"benchmark": benchmark(), "transacional": transacional(), "conjunto_excluido": conjunto_excluido()}
    for nome, bloco in resultado.items():
        print(f"\n{nome}")
        for chave in ("amostra", "premissas", "calendario"):
            if chave in bloco:
                print(f"  {chave}: {bloco[chave]}")
    save_json(resultado, "caracterizacao")
    print("\nresultados gravados em results/")


if __name__ == "__main__":
    main()
