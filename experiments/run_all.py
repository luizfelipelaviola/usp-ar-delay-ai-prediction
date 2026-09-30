"""Executa os tres scripts do trabalho, em ordem, e falha se algum falhar.

    python experiments/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

ETAPAS = [
    ("caracterizacao.py", "Caracterizacao dos conjuntos de dados"),
    ("comparacao.py", "Comparacao entre os tres comparadores"),
    ("figuras.py", "Figuras"),
]


def main() -> None:
    falhas = []
    for script, descricao in ETAPAS:
        print(f"{'=' * 78}\n{descricao}\n{'=' * 78}")
        resultado = subprocess.run([sys.executable, str(RAIZ / "experiments" / script)], cwd=RAIZ)
        if resultado.returncode != 0:
            falhas.append(script)
    if falhas:
        print(f"falharam: {', '.join(falhas)}")
        sys.exit(1)
    print("resultados em results/")


if __name__ == "__main__":
    main()
