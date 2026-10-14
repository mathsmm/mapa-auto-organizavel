"""
Le "lista_filmes_tratada.csv" (titulo;Categoria1|Categoria2|...), codifica cada
filme como um vetor binario de categorias (1 se o filme pertence a categoria,
0 caso contrario) e treina um SOM com o algoritmo de Haykin (som.py).

"lista_filmes_tratada.csv" foi extraida da base de dados ml-latest-small.zip de
https://grouplens.org/datasets/movielens/latest/

Uso:
    python treinar_som.py [--linhas 20] [--colunas 30] [--ordenacao 1000]
                          [--convergencia N] [--semente 42]
"""

import argparse
import os
import time
from collections import Counter
import numpy as np
from som import SOM

PASTA = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_FILMES = os.path.join(PASTA, "lista_filmes_9k.csv")
ARQUIVO_MODELO = os.path.join(PASTA, "m_9k.npz")


def ler_filmes(caminho=ARQUIVO_FILMES):
    titulos, categorias = [], []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.rstrip("\n")
            if not linha:
                continue
            # rsplit: o titulo pode conter ";" mas as categorias nao.
            titulo, cats = linha.rsplit(";", 1)
            titulos.append(titulo)
            categorias.append(cats.split("|"))
    return titulos, categorias


def codificar(categorias):
    """Vetor binário (multi-hot) por filme. Gera uma matriz na forma:
    [[0. 1. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 1. 0. ... 0. 0. 0.]
     ...
     [1. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 1. 0. ... 0. 0. 0.]];
    Categorias ordenadas por frequência."""
    contagem = Counter(c for cats in categorias for c in cats)
    nomes = [c for c, _ in contagem.most_common()]
    indice = {c: k for k, c in enumerate(nomes)}
    dados = np.zeros((len(categorias), len(nomes)))
    for f, cats in enumerate(categorias):
        dados[f, [indice[c] for c in cats]] = 1.0
    return dados, nomes


def treinar(linhas=20, colunas=30, ordenacao=1000, convergencia=None, semente=42,
            saida=ARQUIVO_MODELO):
    titulos, categorias = ler_filmes()
    dados, nomes_categorias = codificar(categorias)
    print(f"{len(titulos)} filmes, {len(nomes_categorias)} categorias")

    som = SOM(linhas, colunas, dados.shape[1], semente=semente)
    t0 = time.perf_counter()
    som.treinar(dados, iter_ordenacao=ordenacao, iter_convergencia=convergencia)
    print(f"Treinamento concluido em {time.perf_counter() - t0:.1f} s")

    som.salvar(saida, categorias=np.array(nomes_categorias))
    print(f"Modelo salvo em {saida}")
    return som


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treina um SOM (Haykin, cap. 9) com a lista de filmes.")
    ap.add_argument("--linhas", type=int, default=20)
    ap.add_argument("--colunas", type=int, default=30)
    ap.add_argument("--ordenacao", type=int, default=1000,
                    help="iteracoes da fase de ordenacao (Haykin: ~1000 ou mais)")
    ap.add_argument("--convergencia", type=int, default=None,
                    help="iteracoes da fase de convergencia (padrao: 500 x numero de neuronios)")
    ap.add_argument("--semente", type=int, default=42)
    args = ap.parse_args()
    treinar(args.linhas, args.colunas, args.ordenacao, args.convergencia, args.semente)
