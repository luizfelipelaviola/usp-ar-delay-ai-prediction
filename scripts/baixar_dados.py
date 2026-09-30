"""Baixa os tres conjuntos de dados das fontes publicas e confere o SHA-256 de cada um.

Os arquivos nao sao redistribuidos neste repositorio. O script os obtem das
fontes e recusa qualquer arquivo diferente do usado no trabalho.

    python scripts/baixar_dados.py
"""

from __future__ import annotations

import hashlib
import http.client
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DESTINO = Path(__file__).resolve().parent.parent / "data" / "raw"

#: (arquivo local, url, membro do zip ou None, sha256 esperado, origem)
FONTES = [
    ("WA_Fn-UseC_-Accounts-Receivable.csv",
     "https://www.kaggle.com/api/v1/datasets/download/hhenry/finance-factoring-ibm-late-payment-histories",
     "WA_Fn-UseC_-Accounts-Receivable.csv",
     "651bc4225708bf33148a0e177c9221afdf697d3a4de10333725a4af3dd022fcf",
     "Kaggle hhenry/finance-factoring-ibm-late-payment-histories"),
    ("transacional.csv",
     "https://raw.githubusercontent.com/SkywalkerHub/Payment-Date-Prediction/"
     "5758490d89867c68cff428658e43c91d57004d92/Dataset.csv",
     None,
     "4f2d7db57c0cc66aee5e44522d5554689c036e5fccd2980192b1af148d501590",
     "GitHub SkywalkerHub/Payment-Date-Prediction, commit 5758490"),
    ("kaggle_ar_delay.csv",
     "https://www.kaggle.com/api/v1/datasets/download/sonalisingh1411/accounts-receivable-and-payment-delay-analysis",
     "Dataset.csv",
     "2a7f1dfc4e52231d4a4cb60ee019d6cef3730708afffb1c12786b4d3061bb32d",
     "Kaggle sonalisingh1411/accounts-receivable-and-payment-delay-analysis"),
]


def baixar(url: str, tentativas: int = 4) -> bytes:
    """Baixa o conteudo, repetindo quando a conexao cai no meio da transferencia."""
    requisicao = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for tentativa in range(1, tentativas + 1):
        try:
            with urllib.request.urlopen(requisicao, timeout=180) as resposta:
                return resposta.read()
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError) as erro:
            if tentativa == tentativas:
                raise
            print(f"       tentativa {tentativa} falhou ({type(erro).__name__}); repetindo")
            time.sleep(5 * tentativa)
    raise RuntimeError("inalcancavel")


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    falhas = 0
    for nome, url, membro, esperado, origem in FONTES:
        conteudo = baixar(url)
        if membro is not None:
            with zipfile.ZipFile(io.BytesIO(conteudo)) as pacote:
                conteudo = pacote.read(membro)
        obtido = hashlib.sha256(conteudo).hexdigest()
        if obtido != esperado:
            print(f"FALHA  {nome}: sha256 {obtido} difere do esperado {esperado}")
            falhas += 1
            continue
        (DESTINO / nome).write_bytes(conteudo)
        print(f"ok     {nome}  ({origem})")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
