"""Caminhos, semente e parametros comuns a todos os experimentos."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

for _d in (RESULTS, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

AR_CSV = DATA_RAW / "WA_Fn-UseC_-Accounts-Receivable.csv"

# Semente usada nos modelos e nas reamostragens do bootstrap.
SEED = 42

# Repeticoes usadas nos intervalos de confianca por bootstrap.
N_BOOTSTRAP = 2000

# Qualquer pagamento depois do vencimento conta como atraso (dias de atraso > 0).
LATE_THRESHOLD_DAYS = 0

# Fracao das datas de emissao reservada ao teste do holdout temporal. No
# benchmark, 0.25 deixa cerca de seis meses de faturas no teste.
TEST_FRACTION = 0.25
