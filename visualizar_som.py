"""
Visualizacao interativa do SOM treinado com a lista de filmes.

Cada hexagono e um neuronio da grade; os filmes aparecem como pontos dentro do
hexagono do seu neuronio vencedor (os mais tipicos no centro). Ao passar o
mouse sobre o mapa, mostra o filme mais proximo do ponto exato do cursor
(titulo e categorias).

Uso:
    python visualizar_som.py              # carrega som_treinado.npz (treina se nao existir)
    python visualizar_som.py --retreinar  # treina novamente antes de abrir
"""

import argparse
import math
import os
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import colormaps

from som import SOM
from treinar_som import ARQUIVO_MODELO, codificar, ler_filmes, treinar

# 20 cores bem distintas (Sasha Trubetskoy), na ordem de frequência das categorias.
PALETA = [
    "#e6194b", "#4363d8", "#3cb44b", "#f58231", "#911eb4", "#ffe119", "#46f0f0",
    "#f032e6", "#9a6324", "#bcf60c", "#008080", "#800000", "#fabebe", "#000075",
    "#808000", "#e6beff", "#aaffc3", "#ffd8b1", "#808080", "#fffac8",
]

MODOS = [
    "Categoria dominante",
    "Mistura de categorias",
    "U-Matrix (distancia entre vizinhos)",
    "Densidade de filmes",
    "Plano de componente (categoria)",
]

R_HEX = 1 / math.sqrt(3)  # raio do hexágono em unidades da grade (vizinhos a distância 1)
R_FILMES = 0.42  # raio da espiral de filmes dentro de cada hexágono
ANGULO_DOURADO = math.pi * (3 - math.sqrt(5))
MARGEM = 12


def hex_para_rgb(cor):
    return np.array([int(cor[k:k + 2], 16) for k in (1, 3, 5)]) / 255.0


def rgb_para_hex(rgb):
    r, g, b = (int(round(255 * min(max(c, 0.0), 1.0))) for c in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def luminancia(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


class Visualizador:
    def __init__(self, raiz, som, titulos, categorias_filmes, nomes_categorias, dados):
        self.raiz = raiz
        self.som = som
        self.titulos = titulos
        self.categorias_filmes = categorias_filmes
        self.nomes = nomes_categorias
        self.paleta = np.array([hex_para_rgb(c) for c in PALETA[:len(nomes_categorias)]])
        self.filmes_por_cat = dados.sum(axis=0)

        # Neurônio vencedor (Eq. 9.3) e erro de quantização de cada filme.
        d2 = som.distancias(dados)
        self.vencedor = np.argmin(d2, axis=1)
        erro = np.sqrt(d2[np.arange(len(dados)), self.vencedor])
        self.contagem = np.bincount(self.vencedor, minlength=som.l)
        self.matriz_u = som.matriz_u()

        # Posição de cada filme no plano do mapa: espiral (girassol) dentro do
        # hexágono do vencedor, com os filmes de menor erro mais ao centro.
        self.pos_filmes = np.zeros((len(dados), 2))
        ordem = np.lexsort((erro, self.vencedor))
        inicio = 0
        for j in range(som.l):
            m = self.contagem[j]
            if m == 0:
                continue
            idx = ordem[inicio:inicio + m]
            inicio += m
            k = np.arange(m)
            r = R_FILMES * np.sqrt(k / m)
            th = k * ANGULO_DOURADO
            self.pos_filmes[idx] = som.posicoes[j] + np.column_stack([r * np.cos(th), r * np.sin(th)])

        self.modo = tk.StringVar(value=MODOS[0])
        self.categoria = tk.StringVar(value=self.nomes[0])
        self.mostrar_filmes = tk.BooleanVar(value=True)
        self._montar_interface()
        self._redesenhar_pendente = None

    # ------------------------------------------------------------- INTERFACE

    def _montar_interface(self):
        self.raiz.title("Self-Organizing Map de filmes (Haykin, cap. 9)")
        self.raiz.geometry("1400x860")

        barra = ttk.Frame(self.raiz, padding=(8, 6))
        barra.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(barra, text="Coloracao:").pack(side=tk.LEFT)
        cb_modo = ttk.Combobox(barra, textvariable=self.modo, values=MODOS, state="readonly", width=36)
        cb_modo.pack(side=tk.LEFT, padx=(4, 12))
        cb_modo.bind("<<ComboboxSelected>>", lambda e: self._atualizar_cores())
        ttk.Label(barra, text="Categoria:").pack(side=tk.LEFT)
        self.cb_categoria = ttk.Combobox(barra, textvariable=self.categoria, values=self.nomes,
                                         state="disabled", width=20)
        self.cb_categoria.pack(side=tk.LEFT, padx=(4, 12))
        self.cb_categoria.bind("<<ComboboxSelected>>", lambda e: self._atualizar_cores())
        ttk.Checkbutton(barra, text="Mostrar filmes", variable=self.mostrar_filmes,
                        command=self._alternar_filmes).pack(side=tk.LEFT)
        ttk.Label(barra, foreground="#555",
                  text=f"   {len(self.titulos)} filmes  |  grade {self.som.linhas}x{self.som.colunas} "
                       f"({self.som.l} neuronios)  |  erro topografico "
                       f"{self.som.historico[-1][4]:.3f}" if self.som.historico else "").pack(side=tk.LEFT)

        self.info = ttk.Label(self.raiz, padding=(8, 4), anchor="w",
                              text="Passe o mouse sobre o mapa para ver o filme mais proximo.")
        self.info.pack(side=tk.BOTTOM, fill=tk.X)

        corpo = ttk.Frame(self.raiz)
        corpo.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.legenda = tk.Canvas(corpo, width=230, bg="white", highlightthickness=0)
        self.legenda.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas = tk.Canvas(corpo, bg="white", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", self._agendar_redesenho)
        self.canvas.bind("<Motion>", self._mouse_moveu)
        self.canvas.bind("<Leave>", lambda e: self._esconder_destaque())

    # --------------------------------------------------- GEOMETRIA / DESENHO

    def _calcular_escala(self):
        larg, alt = self.canvas.winfo_width(), self.canvas.winfo_height()
        p = self.som.posicoes
        x0, x1 = p[:, 0].min() - 0.5, p[:, 0].max() + 0.5
        y0, y1 = p[:, 1].min() - R_HEX, p[:, 1].max() + R_HEX
        self.escala = min((larg - 2 * MARGEM) / (x1 - x0), (alt - 2 * MARGEM) / (y1 - y0))
        self.desloc = np.array([
            (larg - self.escala * (x1 - x0)) / 2 - self.escala * x0,
            (alt - self.escala * (y1 - y0)) / 2 - self.escala * y0,
        ])

    def _para_tela(self, pts):
        return np.asarray(pts) * self.escala + self.desloc

    def _para_grade(self, x, y):
        return (np.array([x, y]) - self.desloc) / self.escala

    def _vertices_hex(self, j):
        cx, cy = self.som.posicoes[j]
        vs = []
        for k in range(6):
            a = math.radians(60 * k + 30)
            vs.append(self._para_tela([cx + R_HEX * math.cos(a), cy + R_HEX * math.sin(a)]))
        return [c for v in vs for c in v]

    def _agendar_redesenho(self, _evento=None):
        if self._redesenhar_pendente:
            self.raiz.after_cancel(self._redesenhar_pendente)
        self._redesenhar_pendente = self.raiz.after(60, self._redesenhar)

    def _redesenhar(self):
        self._redesenhar_pendente = None
        c = self.canvas
        c.delete("all")
        self._calcular_escala()

        self.itens_hex = [
            c.create_polygon(self._vertices_hex(j), outline="white", width=1, tags=("hex",))
            for j in range(self.som.l)
        ]
        raio = max(1.0, self.escala * 0.025)
        tela = self._para_tela(self.pos_filmes)
        for f, (x, y) in enumerate(tela):
            c.create_oval(x - raio, y - raio, x + raio, y + raio, width=0,
                          tags=("filme", f"n{self.vencedor[f]}"))

        # Destaques (neuronio e filme sob o cursor) e dica flutuante.
        self.destaque_hex = c.create_polygon(0, 0, 0, 0, fill="", outline="black", width=2.5,
                                             state="hidden")
        r = max(4.0, self.escala * 0.08)
        self.r_destaque = r
        self.destaque_filme = c.create_oval(0, 0, 0, 0, outline="black", width=2, state="hidden")
        self.destaque_filme2 = c.create_oval(0, 0, 0, 0, outline="white", width=1, state="hidden")
        self.dica_fundo = c.create_rectangle(0, 0, 0, 0, fill="#fffff0", outline="#333",
                                             state="hidden")
        self.dica_titulo = c.create_text(0, 0, anchor="nw", font=("Segoe UI", 11, "bold"),
                                         width=380, state="hidden")
        self.dica_cats = c.create_text(0, 0, anchor="nw", font=("Segoe UI", 10), fill="#333",
                                       width=380, state="hidden")
        self._atualizar_cores()
        self._alternar_filmes()

    # ----------------------------------------------------------------- CORES

    def _cores_neuronios(self):
        """Devolve (lista de cores RGB por neuronio, descricao da legenda)."""
        w = self.som.pesos
        modo = self.modo.get()
        if modo == "Categoria dominante":
            dom = np.argmax(w, axis=1)
            forca = np.clip(w[np.arange(len(w)), dom], 0, 1)[:, None]
            rgb = 1 - (0.35 + 0.65 * forca) * (1 - self.paleta[dom])
            return rgb, ("categorias", dom)
        if modo == "Mistura de categorias":
            p = np.clip(w, 0, None) ** 2
            p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
            return p @ self.paleta, ("categorias", None)
        if modo.startswith("U-Matrix"):
            return self._mapa_cores(self.matriz_u, "viridis_r", "Distancia media aos vizinhos")
        if modo == "Densidade de filmes":
            return self._mapa_cores(np.log1p(self.contagem), "YlOrRd", "Filmes por neuronio",
                                    rotulos=(0, int(self.contagem.max())))
        k = self.nomes.index(self.categoria.get())
        return self._mapa_cores(np.clip(w[:, k], 0, 1), "YlGnBu",
                                f"Peso de '{self.nomes[k]}'", faixa=(0.0, 1.0))

    def _mapa_cores(self, valores, nome_cmap, titulo, faixa=None, rotulos=None):
        vmin, vmax = faixa if faixa else (float(valores.min()), float(valores.max()))
        t = (valores - vmin) / max(vmax - vmin, 1e-12)
        cmap = colormaps[nome_cmap]
        rotulos = rotulos if rotulos else (f"{vmin:.2f}", f"{vmax:.2f}")
        return cmap(t)[:, :3], ("escala", cmap, titulo, rotulos)

    def _atualizar_cores(self):
        modo = self.modo.get()
        self.cb_categoria.configure(state="readonly" if modo.startswith("Plano") else "disabled")
        rgb, legenda = self._cores_neuronios()
        self.cores_rgb = rgb
        for j, item in enumerate(self.itens_hex):
            self.canvas.itemconfigure(item, fill=rgb_para_hex(rgb[j]))
            cor_ponto = "#ffffff" if luminancia(rgb[j]) < 0.45 else "#1b1b1b"
            self.canvas.itemconfigure(f"n{j}", fill=cor_ponto)
        self._desenhar_legenda(legenda)

    def _desenhar_legenda(self, legenda):
        lg = self.legenda
        lg.delete("all")
        if legenda[0] == "categorias":
            dom = legenda[1]
            lg.create_text(12, 14, anchor="w", font=("Segoe UI", 10, "bold"), text="Categorias")
            for k, nome in enumerate(self.nomes):
                y = 38 + k * 22
                lg.create_rectangle(12, y - 7, 28, y + 7, fill=PALETA[k], outline="#444")
                extra = f"  ({int(self.filmes_por_cat[k])})"
                if dom is not None:
                    extra = f"  ({int(self.filmes_por_cat[k])} filmes, {int(np.sum(dom == k))} neur.)"
                lg.create_text(36, y, anchor="w", font=("Segoe UI", 9), text=nome + extra)
            return
        _, cmap, titulo, (rot_min, rot_max) = legenda
        lg.create_text(12, 14, anchor="w", font=("Segoe UI", 10, "bold"), text=titulo, width=210)
        topo, base = 50, 450
        for y in range(topo, base):
            t = 1 - (y - topo) / (base - topo)
            lg.create_line(20, y, 50, y, fill=rgb_para_hex(cmap(t)))
        lg.create_rectangle(20, topo, 50, base, outline="#444")
        lg.create_text(58, topo, anchor="w", text=str(rot_max), font=("Segoe UI", 9))
        lg.create_text(58, base, anchor="w", text=str(rot_min), font=("Segoe UI", 9))

    def _alternar_filmes(self):
        self.canvas.itemconfigure("filme", state="normal" if self.mostrar_filmes.get() else "hidden")

    # ------------------------------------------------------------------ MOUSE

    def _mouse_moveu(self, ev):
        if not hasattr(self, "escala"):
            return
        p = self._para_grade(ev.x, ev.y)
        f = int(np.argmin(np.sum((self.pos_filmes - p) ** 2, axis=1)))
        j = int(self.vencedor[f])
        c = self.canvas

        c.coords(self.destaque_hex, *self._vertices_hex(j))
        fx, fy = self._para_tela(self.pos_filmes[f])
        r = self.r_destaque
        c.coords(self.destaque_filme, fx - r, fy - r, fx + r, fy + r)
        c.coords(self.destaque_filme2, fx - r + 2, fy - r + 2, fx + r - 2, fy + r - 2)

        c.itemconfigure(self.dica_titulo, text=self.titulos[f])
        c.itemconfigure(self.dica_cats, text=" | ".join(self.categorias_filmes[f]))
        for item in (self.destaque_hex, self.destaque_filme, self.destaque_filme2,
                     self.dica_fundo, self.dica_titulo, self.dica_cats):
            c.itemconfigure(item, state="normal")
            c.tag_raise(item)
        self._posicionar_dica(ev.x, ev.y)  # bbox so existe para itens visiveis

        lin, col = divmod(j, self.som.colunas)
        w = self.som.pesos[j]
        top = np.argsort(w)[::-1][:4]
        pesos = "  ".join(f"{self.nomes[k]} {w[k]:.2f}" for k in top if w[k] >= 0.05)
        self.info.configure(
            text=f"{self.titulos[f]}  [{' | '.join(self.categorias_filmes[f])}]      "
                 f"Neuronio (linha {lin}, coluna {col}): {self.contagem[j]} filmes  -  pesos: {pesos}"
        )

    def _posicionar_dica(self, x, y):
        c = self.canvas
        pad = 6
        c.coords(self.dica_titulo, 0, 0)
        b1 = c.bbox(self.dica_titulo)
        c.coords(self.dica_cats, 0, b1[3] + 2)
        b2 = c.bbox(self.dica_cats)
        larg = max(b1[2], b2[2]) + 2 * pad
        alt = b2[3] + 2 * pad
        dx = x + 16 if x + 16 + larg < c.winfo_width() else x - 16 - larg
        dy = y + 16 if y + 16 + alt < c.winfo_height() else y - 16 - alt
        c.coords(self.dica_fundo, dx, dy, dx + larg, dy + alt)
        c.coords(self.dica_titulo, dx + pad, dy + pad)
        c.coords(self.dica_cats, dx + pad, dy + pad + b1[3] + 2)

    def _esconder_destaque(self):
        for item in (self.destaque_hex, self.destaque_filme, self.destaque_filme2,
                     self.dica_fundo, self.dica_titulo, self.dica_cats):
            self.canvas.itemconfigure(item, state="hidden")


def main():
    ap = argparse.ArgumentParser(description="Visualiza o SOM treinado com a lista de filmes.")
    ap.add_argument("--retreinar", action="store_true", help="treina o SOM antes de visualizar")
    args = ap.parse_args()

    titulos, categorias = ler_filmes()
    dados, nomes = codificar(categorias)

    if args.retreinar or not os.path.exists(ARQUIVO_MODELO):
        treinar()
    som, arq = SOM.carregar(ARQUIVO_MODELO)
    if list(arq["categorias"]) != nomes:
        print("Modelo salvo não corresponde ao arquivo de filmes atual; treinando novamente.")
        treinar()
        som, arq = SOM.carregar(ARQUIVO_MODELO)

    try:  # textos nitidos em telas com escala no Windows
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    raiz = tk.Tk()
    Visualizador(raiz, som, titulos, categorias, nomes, dados)
    raiz.mainloop()


if __name__ == "__main__":
    main()
