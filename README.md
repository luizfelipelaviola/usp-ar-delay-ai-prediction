# Predição de atraso de pagamento em contas a receber

Código e resultados do trabalho de conclusão de curso *Aplicação e avaliação de
técnicas de inteligência artificial na predição de atraso de pagamento em contas
a receber de pequenas e médias empresas*, de Luiz Felipe Laviola, apresentado ao
MBA em Inteligência Artificial e Big Data do ICMC/USP em 2026.

## Como reproduzir

Requer Python 3.12.

```bash
pip install -r requirements.txt
python scripts/baixar_dados.py
python experiments/run_all.py
```

O primeiro script baixa os conjuntos de dados das fontes públicas e confere o
SHA-256 de cada arquivo. O segundo executa os experimentos e grava os resultados
em `results/`.

## Resultados

Precisão média no conjunto de teste.

| Conjunto de dados | Regra simples | Regressão logística | XGBoost |
|---|---|---|---|
| *Benchmark* público | 0,6711 | 0,7453 | 0,7190 |
| Conjunto transacional | 0,7287 | 0,7742 | 0,8105 |

## Licença

O código é distribuído sob a licença MIT. Os dados seguem as condições de suas
fontes.
