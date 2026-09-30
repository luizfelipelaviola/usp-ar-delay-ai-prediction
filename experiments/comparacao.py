"""Comparacao entre os tres comparadores do trabalho.

Produz todos os resultados de modelo apresentados no texto. O protocolo e o
holdout temporal, com treino restrito as faturas cujo desfecho ja era conhecido
na data do corte (`arml.splits.maturar`).

Para cada conjunto, os tres comparadores do texto (regra simples, regressao
logistica e XGBoost) sao avaliados com os mesmos atributos, e cada par recebe dois
testes: bootstrap pareado da precisao media e DeLong da ROC-AUC. A correcao de
Holm e aplicada separadamente a cada metrica, sobre as tres comparacoes de cada
conjunto.

No benchmark, o experimento executa ainda o controle positivo de vazamento com o
proprio XGBoost e a ablacao do historico de pagamento na regressao logistica. No
conjunto transacional, registra tambem onde termina o desfecho observado no
arquivo.
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from arml import data, data_transacional
from arml.delong import delong_test
from arml.evaluate import evaluate_zoo, fit_score, header, save_json, save_table
from arml.features import CATEGORICAS, NUMERICAS, NUMERICAS_LEAKY, build_features, build_features_leaky
from arml.metrics import classification_report, holm_correction, paired_bootstrap_test
from arml.models import model_zoo
from arml.splits import maturar, out_of_time

COMPARADORES = ["historico_cliente", "regressao_logistica", "xgboost"]
HISTORICO = [c for c in NUMERICAS if c.startswith("hist_")]


def particao(df: pd.DataFrame):
    completa = out_of_time(df)
    return completa, maturar(df, completa)


def comparar(nome: str, df: pd.DataFrame) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = build_features(df)
    X = feat[NUMERICAS + CATEGORICAS]
    y = df["y_late"].to_numpy().astype(int)
    completa, split = particao(df)
    y_teste = y[split.teste]
    zoo = model_zoo(NUMERICAS, CATEGORICAS)

    header(f"{nome}  ({split.descricao})")
    print(f"  treino {len(split.treino)} (retiradas {len(completa.treino) - len(split.treino)} "
          f"sem desfecho no corte) | teste {len(split.teste)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tabela, scores = evaluate_zoo(
            {k: v for k, v in zoo.items() if k in COMPARADORES}, X, y, split
        )

    colunas = ["modelo", "roc_auc", "roc_auc_ic_inf", "roc_auc_ic_sup",
               "pr_auc", "pr_auc_ic_inf", "pr_auc_ic_sup"]
    save_table(tabela[colunas], f"comparacao_{nome}_modelos")
    resultado = {
        "particao": split.descricao,
        "corte": str(df["InvoiceDate"].iloc[split.teste].min().date()),
        "n_treino_antes_da_restricao": len(completa.treino),
        "n_treino": len(split.treino),
        "n_treino_retiradas_sem_desfecho": len(completa.treino) - len(split.treino),
        "n_teste": len(split.teste),
        "tabela": tabela[colunas].to_dict(orient="records"),
    }

    pares = list(combinations(COMPARADORES, 2))
    testes_pr = [paired_bootstrap_test(y_teste, scores[b], scores[a], "pr_auc") for a, b in pares]
    testes_roc = [delong_test(y_teste, scores[b], scores[a]) for a, b in pares]
    holm_pr = holm_correction([t["p_valor"] for t in testes_pr])
    holm_roc = holm_correction([t["p_valor"] for t in testes_roc])

    linhas = []
    for (a, b), pr, roc, hpr, hroc in zip(pares, testes_pr, testes_roc, holm_pr, holm_roc):
        linhas.append({
            "modelo_a": b,
            "modelo_b": a,
            "diferenca_pr_auc": pr["diferenca"],
            "pr_auc_ic_inf": pr["ic_inferior"],
            "pr_auc_ic_sup": pr["ic_superior"],
            "p_bootstrap_pr_auc": pr["p_valor"],
            "pr_auc_significativo_holm": hpr["significativo_apos_holm"],
            "diferenca_roc_auc": roc["diferenca"],
            "p_delong_roc_auc": roc["p_valor"],
            "roc_auc_significativo_holm": hroc["significativo_apos_holm"],
        })
        print(f"  {b} menos {a}: AP {pr['diferenca']:+.4f} "
              f"[{pr['ic_inferior']:+.4f}; {pr['ic_superior']:+.4f}] p={pr['p_valor']:.4f} "
              f"Holm {'sim' if hpr['significativo_apos_holm'] else 'nao'} | "
              f"ROC-AUC {roc['diferenca']:+.4f} p={roc['p_valor']:.4f} "
              f"Holm {'sim' if hroc['significativo_apos_holm'] else 'nao'}")

    save_table(pd.DataFrame(linhas), f"comparacao_{nome}_testes")
    resultado["comparacoes"] = linhas
    return resultado


def controle_positivo(df: pd.DataFrame) -> dict:
    y = df["y_late"].to_numpy().astype(int)
    _, split = particao(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = build_features_leaky(df)
    fabrica = model_zoo(NUMERICAS_LEAKY, CATEGORICAS)["xgboost"]
    score = fit_score(fabrica, feat[NUMERICAS_LEAKY + CATEGORICAS], y, split)["y_score"]
    m = classification_report(y[split.teste], score)
    print(f"\n  controle positivo (XGBoost com colunas posteriores ao desfecho): "
          f"ROC-AUC {m['roc_auc']:.4f}  AP {m['pr_auc']:.4f}")
    return {"modelo": "xgboost", "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"]}


def ablacao_historico(df: pd.DataFrame) -> dict:
    y = df["y_late"].to_numpy().astype(int)
    _, split = particao(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = build_features(df)
    restantes = [c for c in NUMERICAS if c not in HISTORICO]
    resultado = {}
    for rotulo, colunas in (("completo", NUMERICAS), ("sem_historico_pagamento", restantes)):
        fabrica = model_zoo(colunas, CATEGORICAS)["regressao_logistica"]
        score = fit_score(fabrica, feat[colunas + CATEGORICAS], y, split)["y_score"]
        resultado[rotulo] = classification_report(y[split.teste], score)["pr_auc"]
    print(f"  ablacao, regressao logistica: AP {resultado['completo']:.4f} com historico, "
          f"{resultado['sem_historico_pagamento']:.4f} sem")
    return resultado


def fim_do_desfecho(completo: pd.DataFrame) -> dict:
    observacao = completo.attrs["data_observacao"]
    abertas = completo[completo["evento"] == 0]
    liquidadas = completo[completo["evento"] == 1]
    ultima_liquidada = liquidadas["InvoiceDate"].max()
    return {
        "data_observacao": str(observacao.date()),
        "faturas_em_aberto": int(len(abertas)),
        "primeira_emissao_em_aberto": str(abertas["InvoiceDate"].min().date()),
        "ultima_emissao_liquidada": str(ultima_liquidada.date()),
        "ultima_emissao_do_arquivo": str(completo["InvoiceDate"].max().date()),
        "liquidadas_emitidas_apos_primeira_em_aberto": int(
            (liquidadas["InvoiceDate"] > abertas["InvoiceDate"].min()).sum()),
    }


def main() -> None:
    header("COMPARACAO ENTRE OS TRES COMPARADORES")
    df_benchmark = data.load_dataset()
    benchmark = comparar("benchmark", df_benchmark)
    benchmark["controle_positivo"] = controle_positivo(df_benchmark)
    benchmark["ablacao"] = ablacao_historico(df_benchmark)

    completo = data_transacional.load_dataset()
    liquidadas = completo[completo["evento"] == 1].reset_index(drop=True)
    transacional = comparar("transacional", liquidadas)
    transacional["fim_do_desfecho"] = fim_do_desfecho(completo)
    print(f"\n  fim do desfecho observado: {transacional['fim_do_desfecho']}")

    save_json({"benchmark": benchmark, "transacional": transacional}, "comparacao")
    print("\nresultados gravados em results/")


if __name__ == "__main__":
    main()
