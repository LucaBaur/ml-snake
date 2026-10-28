"""Neuroevolution mit Live-Visualisierung.

Training läuft in einem Hintergrund-Thread, das Fenster spielt immer das aktuell
beste Netz ab und zeigt Neuronen, Gewichte, Aktivierungsfunktion und Mutation.

Tasten:  ↑/↓ Mutationsstärke (sigma)   ←/→ Mutationsrate   1/2/3 Tempo   ESC Ende
"""
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import pygame

from env import SnakeEnv
from nn import (B1_AT, B2_AT, MLP, N_HID, N_IN, N_OUT, N_PARAMS, W1_AT, W2_AT,
                random_params)

POP, ELITE, GAMES, BOARD = 120, 2, 6, 20

# Gemeinsame Einstellungen: UI ändert sie, Trainings-Thread liest sie jede Generation
settings = {"sigma": 0.3, "rate": 0.10}
lock = threading.Lock()
# "elite" = bestes Netz überhaupt, "child" = bestes Netz, das in dieser Gen mutiert wurde
shared = {"gen": -1, "elite": None, "child": None,
          "hist_max": [], "hist_avg": [], "sigma": 0.3, "rate": 0.1}


def evaluate(params, seed0, n_hid=N_HID, games=GAMES):
    """Spielt `games` Spiele, gibt (Fitness, Ø-Score, bester Score) zurück."""
    net, fit, total, best = MLP(params, n_hid), 0.0, 0, 0
    for g in range(games):
        env = SnakeEnv(BOARD, seed=seed0 + g)
        state, done = env.state(), False
        while not done:
            state, _, done = env.step(net.act(state))
        fit += env.score * 100 + min(env.steps, 500) * 0.05
        total += env.score
        best = max(best, env.score)
    return fit / games, total / games, best


def next_generation(pop, rng, sigma, rate):
    """Erzeugt die nächste Generation aus einer nach Fitness SORTIERTEN Population.

    Elite bleibt unverändert, der Rest ist Kreuzung (Crossover) + Mutation.
    Gibt (neue Population, Mutationsmasken) zurück.
    """
    n_par = pop.shape[1]
    parents = pop[: len(pop) // 5]  # Selektion: beste 20 %
    new, new_mut = [pop[i] for i in range(ELITE)], [np.zeros(n_par, bool)] * ELITE
    while len(new) < len(pop):
        a, b = parents[rng.integers(len(parents), size=2)]
        child = np.where(rng.random(n_par) < 0.5, a, b)  # Crossover
        m = rng.random(n_par) < rate                      # welche Gewichte?
        child[m] += rng.normal(0.0, sigma, m.sum())       # um wie viel?
        new.append(child)
        new_mut.append(m)
    return np.array(new), np.array(new_mut)


def train():
    rng = np.random.default_rng()
    pop = random_params(rng, POP)
    mut = np.ones((POP, N_PARAMS), dtype=bool)  # Gen 0: ALLES ist Zufall
    hist_max, hist_avg, gen = [], [], 0
    # Parallelisierung: jedes Netz ist unabhängig -> auf alle CPU-Kerne verteilen
    pool = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))
    while True:
        seed0 = gen * 10  # alle Netze einer Gen spielen dieselben Spiele (fairer Vergleich)
        try:
            res = list(pool.map(evaluate, pop, repeat(seed0), chunksize=8))
        except RuntimeError:  # Programm wird beendet, Pool ist schon geschlossen
            return
        fit = np.array([r[0] for r in res])
        scores = np.array([r[2] for r in res])  # bester Score aus den Spielen
        order = np.argsort(-fit)
        pop, mut, fit, scores = pop[order], mut[order], fit[order], scores[order]
        hist_max.append(int(scores.max()))
        hist_avg.append(float(scores.mean()))
        # bestes Netz, bei dem wirklich etwas mutiert wurde (Elite hat leere Maske)
        c = next((i for i in range(POP) if mut[i].any()), 0)
        pack = lambda i: dict(params=pop[i].copy(), mut=mut[i].copy(), fit=float(fit[i]),
                              score=int(scores[i]), rank=i + 1)
        with lock:
            sigma, rate = settings["sigma"], settings["rate"]
            shared.update(gen=gen, elite=pack(0), child=pack(c),
                          hist_max=list(hist_max), hist_avg=list(hist_avg),
                          sigma=sigma, rate=rate)

        pop, mut = next_generation(pop, rng, sigma, rate)
        gen += 1


# ---------------------------------------------------------------- Darstellung
BG, PANEL, LINE = (18, 18, 24), (26, 26, 36), (60, 60, 78)
POS, NEG, TXT, DIM, YEL = (90, 160, 255), (255, 110, 90), (225, 225, 235), (140, 140, 160), (255, 214, 64)
IN_NAMES = ["Gefahr geradeaus", "Gefahr rechts", "Gefahr links", "Richtung oben",
            "Richtung rechts", "Richtung unten", "Richtung links", "Futter links",
            "Futter rechts", "Futter oben", "Futter unten"]
OUT_NAMES = ["geradeaus", "rechts", "links"]


def mix(c, k, base=BG):
    """Farbe c mit Intensität k (0..1) über dem Hintergrund (ersetzt Alpha)."""
    return tuple(int(base[i] + (c[i] - base[i]) * k) for i in range(3))


class Ui:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1440, 860))
        pygame.display.set_caption("ML-Snake – Neuroevolution")
        self.f = pygame.font.SysFont("consolas", 15)
        self.fs = pygame.font.SysFont("consolas", 12)
        self.fb = pygame.font.SysFont("consolas", 18, bold=True)
        self.clock = pygame.time.Clock()
        self.fps = 12
        self.env = SnakeEnv(BOARD)
        self.net, self.gen, self.mut = None, -1, None
        self.state, self.acts, self.out, self.action = None, None, None, 0
        self.played = 0
        self.mode = "auto"  # auto = abwechselnd Elite / Kind, TAB schaltet um
        self.turn = 0
        self.who = "elite"

    def text(self, s, pos, font=None, color=TXT):
        self.screen.blit((font or self.f).render(s, True, color), pos)

    def box(self, x, y, w, h, title):
        pygame.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=8)
        self.text(title, (x + 10, y + 6), self.fb)

    # -- Spiel
    def load_latest(self):
        self.turn += 1
        self.who = {"elite": "elite", "child": "child"}.get(self.mode, "elite" if self.turn % 2 else "child")
        with lock:
            if shared["elite"] is None:
                return False
            ind = shared[self.who]
            self.net = MLP(ind["params"].copy())
            self.gen, self.mut = shared["gen"], ind["mut"].copy()
            self.snap = {k: shared[k] for k in ("hist_max", "hist_avg", "sigma", "rate")}
            self.snap.update(fit=ind["fit"], score=ind["score"], rank=ind["rank"])
        self.state, self.played = self.env.reset(), self.played + 1
        return True

    def step_game(self):
        if self.net is None and not self.load_latest():
            return
        self.inp = self.state  # was ins Netz geht (für die Anzeige der Eingangsneuronen)
        self.out, self.acts = self.net.forward(self.state)
        self.action = int(np.argmax(self.out))
        self.state, _, done = self.env.step(self.action)
        if done:
            self.load_latest()

    def draw_game(self):
        c, x0, y0 = 20, 10, 50
        self.box(x0 - 4, y0 - 44, 8 * c + 400 + 8 - 8 * c + 0, 520, "Spiel")
        for x in range(BOARD):
            for y in range(BOARD):
                pygame.draw.rect(self.screen, (32, 32, 44), (x0 + x * c, y0 + y * c, c, c), 1)
        for i, (x, y) in enumerate(self.env.snake):
            pygame.draw.rect(self.screen, (140, 255, 170) if i == 0 else (80, 200, 120),
                             (x0 + x * c + 1, y0 + y * c + 1, c - 2, c - 2), border_radius=4)
        fx, fy = self.env.food
        pygame.draw.circle(self.screen, (240, 80, 80), (x0 + fx * c + c // 2, y0 + fy * c + c // 2), c // 2 - 3)
        s = self.snap if self.net is not None else None
        y = y0 + BOARD * c + 12
        label = "ELITE (unverändert)" if self.who == "elite" else f"MUTIERTES KIND (Rang {s['rank']})"
        if self.gen == 0:
            label = "Zufalls-Netz (Gen 0)"
        self.text(f"Gen {self.gen}: {label}", (x0, y), self.f, YEL if self.who == "child" else TXT)
        self.text(f"Score jetzt: {self.env.score}   Schritte: {self.env.steps}", (x0, y + 22))
        if s:
            self.text(f"Im Training: {s['score']} Futter (Ø über {GAMES} Spiele)", (x0, y + 44))
            self.text(f"Fitness: {s['fit']:.1f}   Ansicht: {self.mode} (TAB)", (x0, y + 66))
        self.text(f"{N_PARAMS} Parameter = {N_IN * N_HID + N_HID * N_OUT} Gewichte", (x0, y + 98), self.fs, DIM)
        self.text(f"+ {N_HID + N_OUT} Biases (Schwellen)", (x0, y + 114), self.fs, DIM)
        self.text(f"Zufalls-Baseline: Ø 0.17 Futter", (x0, y + 130), self.fs, DIM)

    # -- Netz
    def draw_network(self):
        X0, Y0, W, H = 440, 40, 610, 510
        self.box(X0, Y0 - 34, W, H + 34 - 0, "Das Netz: 11 Eingänge → 12 Neuronen → 3 Ausgänge")
        xs = (X0 + 175, X0 + 360, X0 + 520)

        def ys(n, top=Y0 + 50, bot=Y0 + H - 28):
            return [(top + bot) / 2] if n == 1 else [top + i * (bot - top) / (n - 1) for i in range(n)]

        yi, yh, yo = ys(N_IN), ys(N_HID), ys(N_OUT, Y0 + 120, Y0 + H - 120)
        p, m = self.net.params, self.mut
        # Verbindungen: Farbe = Vorzeichen, Dicke/Helligkeit = Betrag, gelb = mutiert
        for layer, (n_a, n_b, ya, yb, at, xa, xb) in enumerate(
                [(N_IN, N_HID, yi, yh, W1_AT, xs[0], xs[1]), (N_HID, N_OUT, yh, yo, W2_AT, xs[1], xs[2])]):
            for a in range(n_a):
                for b in range(n_b):
                    k = at + a * n_b + b
                    w, mutated = p[k], m[k]
                    base = POS if w > 0 else NEG
                    inten = min(1.0, abs(w) / 1.5)
                    if mutated and not m.all():
                        pygame.draw.line(self.screen, YEL, (xa, ya[a]), (xb, yb[b]), 4)
                    pygame.draw.line(self.screen, mix(base, 0.12 + 0.88 * inten, PANEL),
                                     (xa, ya[a]), (xb, yb[b]), 1 + int(inten * 2))
        # Neuronen: Füllung = aktuelle Aktivierung
        z_h, a_h = self.acts
        for i in range(N_IN):
            v = float(self.inp[i])
            pygame.draw.circle(self.screen, mix((235, 235, 245), 0.15 + 0.85 * v, PANEL), (xs[0], yi[i]), 9)
            pygame.draw.circle(self.screen, LINE, (xs[0], yi[i]), 9, 1)
            self.text(IN_NAMES[i], (xs[0] - 22 - 8 * len(IN_NAMES[i]) * 0.78, yi[i] - 8), self.fs, TXT if v else DIM)
        for j in range(N_HID):
            a = float(a_h[j])
            pygame.draw.circle(self.screen, mix(POS if a > 0 else NEG, 0.1 + 0.9 * abs(a), PANEL), (xs[1], yh[j]), 11)
            ring = YEL if (m[B1_AT + j] and not m.all()) else LINE
            pygame.draw.circle(self.screen, ring, (xs[1], yh[j]), 11, 3 if ring == YEL else 1)
        best = int(np.argmax(self.out))
        for k in range(N_OUT):
            pygame.draw.circle(self.screen, mix(TXT, 0.9 if k == best else 0.2, PANEL), (xs[2], yo[k]), 15)
            ring = YEL if (m[B2_AT + k] and not m.all()) else LINE
            pygame.draw.circle(self.screen, ring, (xs[2], yo[k]), 15, 3 if ring == YEL else 1)
            self.text(f"{OUT_NAMES[k]}", (xs[2] + 20, yo[k] - 16), self.f, TXT if k == best else DIM)
            self.text(f"{self.out[k]:+.2f}", (xs[2] + 20, yo[k] + 2), self.fs, TXT if k == best else DIM)
        self.text("Eingabe", (xs[0] - 20, Y0 + H - 14), self.fs, DIM)
        self.text("Versteckte Schicht", (xs[1] - 60, Y0 + H - 14), self.fs, DIM)
        self.text("Entscheidung", (xs[2] - 25, Y0 + H - 14), self.fs, DIM)
        # Legende
        lx, ly = X0 + 12, Y0 + 2
        pygame.draw.line(self.screen, POS, (lx, ly + 22), (lx + 22, ly + 22), 3)
        self.text("Gewicht > 0 (verstärkt)", (lx + 28, ly + 14), self.fs)
        pygame.draw.line(self.screen, NEG, (lx + 200, ly + 22), (lx + 222, ly + 22), 3)
        self.text("< 0 (hemmt)", (lx + 228, ly + 14), self.fs)
        pygame.draw.line(self.screen, YEL, (lx + 320, ly + 22), (lx + 342, ly + 22), 4)
        self.text("= zufällig mutiert", (lx + 348, ly + 14), self.fs)

    # -- Aktivierungsfunktion
    def draw_activation(self):
        X, Y, W, H = 440, 576, 300, 270
        self.box(X, Y, W, H, "Aktivierungsfunktion tanh")
        gx, gy, gw, gh = X + 24, Y + 40, W - 48, 140
        pygame.draw.rect(self.screen, BG, (gx, gy, gw, gh))
        pygame.draw.line(self.screen, LINE, (gx, gy + gh / 2), (gx + gw, gy + gh / 2))
        pygame.draw.line(self.screen, LINE, (gx + gw / 2, gy), (gx + gw / 2, gy + gh))
        to = lambda z, a: (gx + (np.clip(z, -4, 4) + 4) / 8 * gw, gy + gh / 2 - a * gh / 2 * 0.95)
        pts = [to(z, np.tanh(z)) for z in np.linspace(-4, 4, 80)]
        pygame.draw.lines(self.screen, DIM, False, pts, 2)
        z_h, a_h = self.acts
        for z, a in zip(z_h, a_h):
            pygame.draw.circle(self.screen, POS if a > 0 else NEG, to(z, np.tanh(z)), 5)
        self.text("z (Summe)", (gx + gw - 70, gy + gh + 2), self.fs, DIM)
        self.text("a", (gx + gw / 2 + 5, gy - 2), self.fs, DIM)
        for i, s in enumerate(["Jeder Punkt = 1 Neuron: z = Σ(w·x)+b,",
                               "dann a = tanh(z) quetscht in (-1,1).",
                               "Ohne Knick (nichtlinear) wäre das",
                               "ganze Netz nur eine Matrix-Rechnung."]):
            self.text(s, (X + 12, Y + 192 + i * 17), self.fs, DIM)

    # -- Mutation
    def draw_mutation(self):
        X, Y, W, H = 760, 576, 290, 270
        self.box(X, Y, W, H, "Zufall & Mutation")
        s = self.snap if self.net is not None else {"sigma": settings["sigma"], "rate": settings["rate"]}
        sg = settings["sigma"]
        gx, gy, gw, gh = X + 20, Y + 40, W - 40, 90
        pygame.draw.rect(self.screen, BG, (gx, gy, gw, gh))
        pts = []
        for i in range(120):
            d = -2 + 4 * i / 119  # Änderung am Gewicht
            pts.append((gx + i / 119 * gw, gy + gh - 8 - 70 * np.exp(-0.5 * (d / sg) ** 2)))
        pygame.draw.lines(self.screen, YEL, False, pts, 2)
        pygame.draw.line(self.screen, LINE, (gx + gw / 2, gy), (gx + gw / 2, gy + gh))
        self.text("-2", (gx, gy + gh + 2), self.fs, DIM)
        self.text("Änderung des Gewichts", (gx + gw / 2 - 70, gy + gh + 2), self.fs, DIM)
        self.text("+2", (gx + gw - 20, gy + gh + 2), self.fs, DIM)
        n_mut = int(self.mut.sum())
        n_total = self.mut.size
        if self.gen == 0:
            info = "Gen 0: ALLE Gewichte komplett zufällig!"
        else:
            info = f"{'Elite' if self.who == 'elite' else 'Kind'}: {n_mut}/{n_total} Zahlen mutiert"
        self.text(info, (X + 12, Y + 150), self.f, YEL)
        for i, line in enumerate([f"Mutationsrate: {settings['rate'] * 100:.0f}% der Gewichte (←/→)",
                                  f"Stärke sigma: {sg:.2f}  (↑/↓)",
                                  "Glocke = Verteilung der Änderung:",
                                  "schmal = fein, breit = wild.",
                                  "Elite (2 beste) bleibt unverändert."]):
            self.text(line, (X + 12, Y + 176 + i * 17), self.fs, TXT if i < 2 else DIM)

    # -- Lernkurve + Erklärung
    def draw_curve(self):
        X, Y, W, H = 1070, 6, 350, 330
        self.box(X, Y, W, H, "Lernkurve (Futter pro Gen)")
        hm, ha = self.snap["hist_max"], self.snap["hist_avg"]
        gx, gy, gw, gh = X + 36, Y + 40, W - 52, H - 90
        pygame.draw.rect(self.screen, BG, (gx, gy, gw, gh))
        top = max(10, max(hm) + 2)
        n = max(len(hm), 2)
        for series, col in ((ha, DIM), (hm, (140, 255, 170))):
            pts = [(gx + i / (n - 1) * gw, gy + gh - v / top * gh) for i, v in enumerate(series)]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, col, False, pts, 2)
        self.text(str(top), (X + 8, gy - 4), self.fs, DIM)
        self.text("0", (X + 20, gy + gh - 10), self.fs, DIM)
        self.text(f"Generation 0 … {len(hm) - 1}", (gx, gy + gh + 4), self.fs, DIM)
        self.text(f"grün: bestes Netz ({hm[-1]})   grau: Ø ({ha[-1]:.1f})", (gx, gy + gh + 22), self.fs)

    def draw_help(self):
        X, Y, W, H = 1070, 350, 350, 496
        self.box(X, Y, W, H, "Was passiert hier?")
        lines = [
            ("1. Zufall", YEL), ("120 Netze mit Zufallsgewichten spielen.", TXT),
            ("Die meisten sterben sofort.", DIM), ("", TXT),
            ("2. Auslese (Selektion)", YEL), ("Fitness = Futter*100 + Überleben.", TXT),
            ("Die besten 20% werden Eltern.", DIM), ("", TXT),
            ("3. Kreuzung (Crossover)", YEL), ("Kind nimmt je Gewicht den Wert", TXT),
            ("von Elternteil A oder B.", DIM), ("", TXT),
            ("4. Mutation", YEL), ("Einzelne Gewichte werden um einen", TXT),
            ("Zufallswert verschoben (Glocke).", DIM), ("", TXT),
            ("5. Wiederholen", YEL), ("Kein Gradient, keine Formel für", TXT),
            ("'richtig': nur Ausprobieren und", DIM), ("Behalten, was funktioniert.", DIM),
            ("", TXT), ("Tempo: 1 / 2 / 3     Ende: ESC", DIM),
        ]
        for i, (s, c) in enumerate(lines):
            self.text(s, (X + 14, Y + 40 + i * 20), self.f, c)

    def run(self, max_frames=None, shot=None):
        frame, running = 0, True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_TAB:
                        self.mode = {"auto": "elite", "elite": "child", "child": "auto"}[self.mode]
                    with lock:
                        if e.key == pygame.K_UP:
                            settings["sigma"] = min(1.5, settings["sigma"] * 1.25)
                        elif e.key == pygame.K_DOWN:
                            settings["sigma"] = max(0.02, settings["sigma"] / 1.25)
                        elif e.key == pygame.K_RIGHT:
                            settings["rate"] = min(0.5, settings["rate"] + 0.02)
                        elif e.key == pygame.K_LEFT:
                            settings["rate"] = max(0.01, settings["rate"] - 0.02)
                    self.fps = {pygame.K_1: 6, pygame.K_2: 20, pygame.K_3: 80}.get(e.key, self.fps)

            self.screen.fill(BG)
            self.step_game()
            if self.net is None or self.acts is None:
                self.text("Generation 0 wird berechnet …", (500, 400), self.fb)
            else:
                self.draw_game()
                self.draw_network()
                self.draw_activation()
                self.draw_mutation()
                self.draw_curve()
                self.draw_help()
            pygame.display.flip()
            self.clock.tick(self.fps)
            frame += 1
            if max_frames and frame >= max_frames:
                if shot:
                    pygame.image.save(self.screen, shot)
                running = False
        pygame.quit()


if __name__ == "__main__":
    threading.Thread(target=train, daemon=True).start()
    if "--smoke" in sys.argv:  # Test ohne Benutzer: Screenshot nach einigen Sekunden
        import os
        import time
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        time.sleep(float(sys.argv[sys.argv.index("--smoke") + 1]))
        Ui().run(max_frames=30, shot=sys.argv[sys.argv.index("--smoke") + 2])
    else:
        Ui().run()
