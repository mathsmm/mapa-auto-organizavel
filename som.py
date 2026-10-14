"""
Self-Organizing Map (SOM) de Kohonen, implementado segundo HAYKIN,
"Neural Networks and Learning Machines", 3a ed., capitulo 9.

Equacoes utilizadas (numeracao do livro):

  (9.1)  x  = [x1, x2, ..., xm]^T                      vetor de entrada
  (9.2)  wj = [wj1, wj2, ..., wjm]^T, j = 1..l          pesos do neuronio j
  (9.3)  i(x) = arg min_j ||x - wj||                    neuronio vencedor
  (9.5)  d2_ji = ||rj - ri||^2                          distancia lateral na grade
  (9.6)  sigma(n) = sigma0 * exp(-n / tau1)             largura da vizinhanca
  (9.7)  h_ji(x)(n) = exp(-d2_ji / (2 sigma(n)^2))            vizinhanca gaussiana
  (9.13) wj(n+1) = wj(n) + eta(n) h_ji(x)(n) (x(n) - wj(n))   atualização do vetor de pesos
  (9.14) eta(n) = eta0 * exp(-n / tau2)                 taxa de aprendizado

Fases do processo adaptativo (secao 9.3):
  1. Ordenacao (auto-organizacao): eta0 = 0.1, tau2 = 1000,
     sigma0 = "raio" da grade, tau1 = 1000 / log(sigma0).
     Ao final desta fase sigma = 1 (apenas os vizinhos mais proximos).
  2. Convergencia: no minimo 500 x (numero de neuronios) iteracoes,
     eta mantido pequeno (~0.01, nunca zero) e a vizinhanca contendo apenas
     os vizinhos mais proximos, podendo encolher ate quase so o vencedor.
"""

import math
import numpy as np


class SOM:
    def __init__(self, linhas, colunas, dim, semente=None):
        self.linhas = linhas
        self.colunas = colunas
        self.dim = dim
        self.l = linhas * colunas
        self.rng = np.random.default_rng(semente)

        # Posicoes rj dos neuronios no espaco de saida discreto (grade hexagonal:
        # linhas impares deslocadas meia unidade. Para cada neurônio, todos os 
        # seus 6 vizinhos estão a 1 unidade de distancia).
        lin, col = np.divmod(np.arange(self.l), colunas)
        self.posicoes = np.column_stack(
            [col + 0.5 * (lin % 2), lin * math.sqrt(3) / 2]
        ).astype(float)

        # Eq. (9.5): d2_ji = ||rj - ri||^2, pre-calculada para todos os pares.
        dif = self.posicoes[:, None, :] - self.posicoes[None, :, :]
        self.dist2_lateral = np.sum(dif**2, axis=2)

        self.pesos = np.zeros((self.l, dim))
        self.historico = []

    # ------------------------------------------------------------------ passos

    def inicializar(self, dados):
        """Passo 1 (Inicializacao): escolhe os wj(0) aleatoriamente entre os
        vetores de entrada disponiveis, para que o mapa inicial ja esteja na
        faixa do mapa final. Um ruido pequeno garante que os wj(0) sejam
        diferentes entre si, mesmo havendo entradas repetidas."""
        idx = self.rng.choice(len(dados), size=self.l, replace=len(dados) < self.l)
        self.pesos = dados[idx].astype(float) + self.rng.normal(0, 0.01, (self.l, self.dim))

    def idx_bmu(self, x):
        """Passo 3 (Casamento por similaridade), Eq. (9.3)."""
        return int(np.argmin(np.sum((self.pesos - x) ** 2, axis=1)))

    def vizinhanca(self, i, sigma):
        """Eq. (9.7): vizinhanca topologica gaussiana centrada no vencedor i."""
        return np.exp(-self.dist2_lateral[i] / (2.0 * sigma**2))

    def atualizar(self, x, i, eta, sigma):
        """Passo 4 (Atualizacao), Eq. (9.13), aplicada a todos os neuronios."""
        h = self.vizinhanca(i, sigma)
        self.pesos += eta * h[:, None] * (x - self.pesos)

    # ---------------------------------------------------------------- treino

    def treinar(
        self,
        dados,
        iter_ordenacao=1000,
        iter_convergencia=None,
        eta0=0.1,
        eta_convergencia=0.01,
        sigma_final=0.5,
        relatar_a_cada=None,
    ):
        dados = np.asarray(dados, dtype=float)
        self.inicializar(dados)

        # Parametros da fase de ordenacao (Haykin, secao 9.3), generalizados
        # para N1 = iter_ordenacao (com N1 = 1000 sao exatamente os do livro).
        n1 = iter_ordenacao
        sigma0 = np.ptp(self.posicoes, axis=0).max() / 2.0
        tau1 = n1 / math.log(sigma0)
        tau2 = float(n1)

        # Fase de convergencia: pelo menos 500 x numero de neuronios.
        n2 = iter_convergencia if iter_convergencia is not None else 500 * self.l
        # sigma encolhe de 1 (vizinhos mais proximos) ate sigma_final.
        tau3 = n2 / math.log(1.0 / sigma_final) if sigma_final < 1 else math.inf

        total = n1 + n2
        relatar_a_cada = relatar_a_cada or max(total // 20, 1)
        amostras = self.rng.integers(0, len(dados), size=total)

        print(f"Grade {self.linhas}x{self.colunas} = {self.l} neuronios, entrada m = {self.dim}")
        print(f"Fase 1 (ordenacao):    {n1} iteracoes, eta0 = {eta0}, tau2 = {tau2:.0f}, "
              f"sigma0 = {sigma0:.2f}, tau1 = {tau1:.1f}")
        print(f"Fase 2 (convergencia): {n2} iteracoes, eta = {eta_convergencia}, "
              f"sigma: 1.0 -> {sigma_final}")

        for n in range(total):
            if n < n1:
                sigma = sigma0 * math.exp(-n / tau1)  # Eq. (9.6)
                eta = eta0 * math.exp(-n / tau2)  # Eq. (9.14)
            else:
                k = n - n1
                sigma = math.exp(-k / tau3)
                eta = eta_convergencia

            x = dados[amostras[n]]  # Passo 2 (Amostragem)
            i = self.idx_bmu(x)  # Passo 3, Eq. (9.3)
            self.atualizar(x, i, eta, sigma)  # Passo 4, Eq. (9.13)

            if (n + 1) % relatar_a_cada == 0 or n + 1 in (n1, total):
                eq = self.erro_quantizacao(dados)
                et = self.erro_topografico(dados)
                self.historico.append((n + 1, eta, sigma, eq, et))
                fase = "ordenacao" if n < n1 else "convergencia"
                print(f"  n = {n + 1:>7}/{total} [{fase:>12}]  eta = {eta:.4f}  "
                      f"sigma = {sigma:6.3f}  erro quant. = {eq:.4f}  erro topog. = {et:.4f}")

    # ------------------------------------------------------- AVALIAÇÃO / USO

    def distancias(self, dados, bloco=2048):
        """Distâncias euclidianas ao quadrado entre cada entrada e cada neuroôio."""
        dados = np.asarray(dados, dtype=float)
        p2 = np.sum(self.pesos**2, axis=1)
        saida = np.empty((len(dados), self.l))
        for a in range(0, len(dados), bloco):
            d = dados[a:a + bloco]
            saida[a:a + bloco] = np.sum(d**2, axis=1)[:, None] - 2 * d @ self.pesos.T + p2
        return np.maximum(saida, 0.0)

    def erro_quantizacao(self, dados):
        """Distância média entre cada entrada e o peso do seu neurônio vencedor."""
        return float(np.mean(np.sqrt(np.min(self.distancias(dados), axis=1))))

    def erro_topografico(self, dados):
        """Proporção de amostras para as quais os dois melhores neurônios não são vizinhos."""
        d = self.distancias(dados)
        dois = np.argpartition(d, 1, axis=1)[:, :2]
        return float(np.mean(self.dist2_lateral[dois[:, 0], dois[:, 1]] > 1.0 + 1e-9))

    def matriz_u(self):
        """U-Matrix: distância média de cada neurônio aos seus vizinhos imediatos."""
        vizinhos = (self.dist2_lateral > 0) & (self.dist2_lateral <= 1.0 + 1e-9)
        dif = np.sqrt(np.sum((self.pesos[:, None, :] - self.pesos[None, :, :]) ** 2, axis=2))
        return np.sum(dif * vizinhos, axis=1) / np.sum(vizinhos, axis=1)

    # ------------------------------------------------------------ PERSISTÊNCIA

    def salvar(self, caminho, **extras):
        np.savez_compressed(
            caminho,
            pesos=self.pesos,
            linhas=self.linhas,
            colunas=self.colunas,
            historico=np.array(self.historico),
            **extras,
        )

    @classmethod
    def carregar(cls, caminho):
        arq = np.load(caminho, allow_pickle=False)
        som = cls(int(arq["linhas"]), int(arq["colunas"]), arq["pesos"].shape[1])
        som.pesos = arq["pesos"]
        som.historico = [tuple(h) for h in arq["historico"]]
        return som, arq
