"""Predicao de atraso de pagamento em contas a receber.

Codigo do trabalho de conclusao de curso. Modulos:

    config             caminhos, semente e parametros globais
    data               carregamento do benchmark publico
    data_transacional  carregamento do conjunto transacional
    data_kaggle_ar     carregamento do conjunto candidato excluido
    caracterizacao     emissao de faturas por dia da semana
    features           atributos restritos ao que era conhecido na emissao
    splits             holdout temporal e restricao do rotulo no corte
    models             os tres comparadores e o pre-processamento
    metrics            precisao media, ROC-AUC, bootstrap e correcao de Holm
    delong             teste de DeLong
    evaluate           ajuste, avaliacao e gravacao dos resultados
"""

__version__ = "1.0.0"
