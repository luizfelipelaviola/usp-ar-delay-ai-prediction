"""Figuras do trabalho, com a distribuicao do atraso e a heterogeneidade entre clientes no benchmark.

Escala de cinza com um acento, legivel impressa em preto e branco.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from arml.config import FIGURES
from arml.data import load_dataset
from arml.evaluate import header

ACENTO = "#1f4e79"
CINZA = "#7a7a7a"
CLARO = "#c8c8c8"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def salvar(fig, nome: str) -> None:
    destino = FIGURES / f"{nome}.png"
    fig.savefig(destino)
    plt.close(fig)
    print(f"  {destino.relative_to(FIGURES.parent.parent)}")


def main() -> None:
    header("FIGURAS")
    df = load_dataset()

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    a.hist(df["y_days"], bins=range(0, 47), color=CLARO, edgecolor=CINZA, linewidth=0.5)
    a.set_xlabel("dias de atraso")
    a.set_ylabel("faturas")
    a.set_title("Todas as faturas", fontsize=9)
    atrasadas = df.loc[df["y_late"] == 1, "y_days"]
    b.hist(atrasadas, bins=range(0, 47), color=ACENTO, alpha=0.85)
    b.set_xlabel("dias de atraso")
    b.set_title(f"Apenas as atrasadas (n = {len(atrasadas)})", fontsize=9)
    fig.suptitle("Distribuição do atraso de pagamento", fontsize=10)
    salvar(fig, "fig01_distribuicao_atraso")

    por_cliente = df.groupby("customerID")["y_late"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(por_cliente)), por_cliente.to_numpy(), color=ACENTO, width=1.0)
    ax.axhline(df["y_late"].mean(), color="black", linestyle="--", linewidth=1,
               label=f"média geral {df['y_late'].mean() * 100:.1f}%".replace(".", ","))
    ax.set_xlabel("clientes, ordenados pela taxa de atraso")
    ax.set_ylabel("taxa de atraso")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor * 100:.0f}%"))
    ax.set_title("Heterogeneidade da taxa de atraso entre clientes", fontsize=10)
    ax.legend(frameon=False)
    salvar(fig, "fig02_heterogeneidade_clientes")


if __name__ == "__main__":
    main()
