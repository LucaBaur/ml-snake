"""Live-Fenster: Ein Netz lernt per Backpropagation, den Lehrer nachzuahmen.

Du siehst: Gewichte (Farbe), welche Gewichte sich gerade ändern (gelb), den Fehler (Loss),
die Trefferquote und ein konkretes Rechenbeispiel für EIN Gewicht.

Tasten:  ↑/↓ Lernrate η   1/2/3 Tempo   SPACE Pause   R Gewichte neu würfeln   ESC Ende
"""
import json
import os
import sys
from collections import deque

import numpy as np
import pygame

from env import SnakeEnv
from evolve import BG, DIM, IN_NAMES, LINE, NEG, OUT_NAMES, PANEL, POS, TXT, YEL, mix
from imitation import BOARD, OUT, Trainer, collect, expert, play
from nn import B1_AT, B2_AT, N_HID, N_IN, N_OUT, W1_AT, W2_AT


def param_name(k):
    """Menschlicher Name für den Parameter mit Index k im flachen Vektor."""
    if k < B1_AT:
        i, j = divmod(k - W1_AT, N_HID)
        return f"Gewicht {IN_NAMES[i]} → Neuron {j + 1}"
    if k < W2_AT:
        return f"Bias von Neuron {k - B1_AT + 1}"
    if k < B2_AT:
        j, o = divmod(k - W2_AT, N_OUT)
        return f"Gewicht Neuron {j + 1} → {OUT_NAMES[o]}"
    return f"Bias von {OUT_NAMES[k - B2_AT]}"


class Live:
    def _init_window(self, caption):
        """Fenster, Schriften, Uhr. Von Unterklassen (z. B. dqn_live.py) wiederverwendet."""
        pygame.init()
        self.screen = pygame.display.set_mode((1440, 860))
        pygame.display.set_caption(caption)
        self.f = pygame.font.SysFont("consolas", 15)
        self.fs = pygame.font.SysFont("consolas", 12)
        self.fb = pygame.font.SysFont("consolas", 18, bold=True)
        self.clock = pygame.time.Clock()

    def __init__(self):
        self._init_window("ML-Snake – Backpropagation")
        X, y = collect(n_games=300)
        self.n_states = len(np.unique(X, axis=0))
        self.t = Trainer(X, y)
        self.expert_score = play(expert, 30)
        self.evo_score = self._evo_reference()
        self.env = SnakeEnv(BOARD)
        self.state = self.env.reset()
        self.inp, self.out, self.acts = self.state, None, None
        self.k, self.paused = 2, False
        self.loss_hist, self.acc_hist = [], []
        self.scores = deque(maxlen=10)
        self.ema = None

    @staticmethod
    def _evo_reference():
        """Ø Test-Score der 12-Neuronen-Evolution aus dem Vergleichsexperiment (falls vorhanden)."""
        try:
            with open(os.path.join(OUT, "hidden_size.json")) as f:
                return float(np.mean(json.load(f)["sizes"]["12"]["tests"]))
        except (OSError, KeyError):
            return None

    def text(self, s, pos, font=None, color=TXT):
        self.screen.blit((font or self.f).render(s, True, color), pos)

    def box(self, x, y, w, h, title):
        pygame.draw.rect(self.screen, PANEL, (x, y, w, h), border_radius=8)
        self.text(title, (x + 10, y + 6), self.fb)

    # ---------------------------------------------------------- Logik
    def train_frame(self):
        if self.paused:
            return
        for _ in range(self.k):
            loss = self.t.step()
            self.ema = loss if self.ema is None else 0.9 * self.ema + 0.1 * loss
            self.loss_hist.append(self.ema)
            if self.t.steps % 10 == 0:
                self.acc_hist.append((self.t.steps, self.t.val_accuracy()))

    def game_frame(self):
        self.inp = self.state
        self.out, self.acts = self.t.net.forward(self.state)
        self.state, _, done = self.env.step(int(np.argmax(self.out)))
        if done:
            self.scores.append(self.env.score)
            self.state = self.env.reset()

    # ---------------------------------------------------------- Zeichnen
    def draw_game(self):
        c, x0, y0 = 20, 10, 50
        self.box(x0 - 4, y0 - 44, 408, 600, "Der Schüler spielt (live)")
        for x in range(BOARD):
            for y in range(BOARD):
                pygame.draw.rect(self.screen, (32, 32, 44), (x0 + x * c, y0 + y * c, c, c), 1)
        for i, (x, y) in enumerate(self.env.snake):
            pygame.draw.rect(self.screen, (140, 255, 170) if i == 0 else (80, 200, 120),
                             (x0 + x * c + 1, y0 + y * c + 1, c - 2, c - 2), border_radius=4)
        fx, fy = self.env.food
        pygame.draw.circle(self.screen, (240, 80, 80), (x0 + fx * c + c // 2, y0 + fy * c + c // 2), c // 2 - 3)
        y = y0 + BOARD * c + 12
        t = self.t
        avg = f"{np.mean(self.scores):.1f}" if self.scores else "–"
        lines = [
            (f"Score jetzt: {self.env.score}   Ø letzte 10 Spiele: {avg}", TXT),
            (f"Lernschritt {t.steps}   Epoche {t.epoch}", TXT),
            (f"Fehler (Loss): {self.ema:.3f}" if self.ema is not None else "Fehler: –", TXT),
            (f"Trefferquote auf Prüfdaten: {self.acc_hist[-1][1]:.1%}" if self.acc_hist else "Trefferquote: –", TXT),
            (f"Lernrate η = {t.lr:.3f} (↑/↓)   Momentum {t.momentum}   Batch {t.batch}", DIM),
            (f"{len(t.X)} Lernbeispiele, aber nur {self.n_states} verschiedene Zustände", DIM),
            ("PAUSE (SPACE)" if self.paused else "", YEL),
        ]
        for i, (s, col) in enumerate(lines):
            self.text(s, (x0, y + i * 22), self.f if i < 4 else self.fs, col)

    def draw_network(self):
        X0, Y0, W, H = 440, 40, 610, 510
        self.box(X0, Y0 - 34, W, H + 34, "Das Netz: Gewichte werden Schritt für Schritt angepasst")
        xs = (X0 + 175, X0 + 360, X0 + 520)

        def ys(n, top=Y0 + 50, bot=Y0 + H - 28):
            return [(top + bot) / 2] if n == 1 else [top + i * (bot - top) / (n - 1) for i in range(n)]

        yi, yh, yo = ys(N_IN), ys(N_HID), ys(N_OUT, Y0 + 120, Y0 + H - 120)
        p, step = self.t.params, self.t.last_step
        rel = np.abs(step) / (np.abs(step).max() + 1e-12)  # 0..1: wie stark ändert sich der Parameter gerade
        for n_a, n_b, ya, yb, at, xa, xb in [(N_IN, N_HID, yi, yh, W1_AT, xs[0], xs[1]),
                                              (N_HID, N_OUT, yh, yo, W2_AT, xs[1], xs[2])]:
            for a in range(n_a):
                for b in range(n_b):
                    k = at + a * n_b + b
                    inten = min(1.0, abs(p[k]) / 1.5)
                    if rel[k] > 0.3 and not self.paused:
                        pygame.draw.line(self.screen, YEL, (xa, ya[a]), (xb, yb[b]), 2 + int(rel[k] * 4))
                    pygame.draw.line(self.screen, mix(POS if p[k] > 0 else NEG, 0.12 + 0.88 * inten, PANEL),
                                     (xa, ya[a]), (xb, yb[b]), 1 + int(inten * 2))
        if self.acts is None:
            return
        a_h = self.acts[1]
        for i in range(N_IN):
            v = float(self.inp[i])
            pygame.draw.circle(self.screen, mix((235, 235, 245), 0.15 + 0.85 * v, PANEL), (xs[0], yi[i]), 9)
            pygame.draw.circle(self.screen, LINE, (xs[0], yi[i]), 9, 1)
            self.text(IN_NAMES[i], (xs[0] - 22 - 8 * len(IN_NAMES[i]) * 0.78, yi[i] - 8), self.fs, TXT if v else DIM)
        for j in range(N_HID):
            a = float(a_h[j])
            pygame.draw.circle(self.screen, mix(POS if a > 0 else NEG, 0.1 + 0.9 * abs(a), PANEL), (xs[1], yh[j]), 11)
            pygame.draw.circle(self.screen, LINE, (xs[1], yh[j]), 11, 1)
        best = int(np.argmax(self.out))
        for k in range(N_OUT):
            pygame.draw.circle(self.screen, mix(TXT, 0.9 if k == best else 0.2, PANEL), (xs[2], yo[k]), 15)
            self.text(OUT_NAMES[k], (xs[2] + 20, yo[k] - 16), self.f, TXT if k == best else DIM)
            self.text(f"{self.out[k]:+.2f}", (xs[2] + 20, yo[k] + 2), self.fs, TXT if k == best else DIM)
        lx, ly = X0 + 12, Y0 + 2
        pygame.draw.line(self.screen, POS, (lx, ly + 22), (lx + 22, ly + 22), 3)
        self.text("Gewicht > 0", (lx + 28, ly + 14), self.fs)
        pygame.draw.line(self.screen, NEG, (lx + 130, ly + 22), (lx + 152, ly + 22), 3)
        self.text("< 0", (lx + 158, ly + 14), self.fs)
        pygame.draw.line(self.screen, YEL, (lx + 210, ly + 22), (lx + 232, ly + 22), 4)
        self.text("= ändert sich gerade stark", (lx + 238, ly + 14), self.fs)

    def draw_step(self):
        X, Y, W, H = 440, 576, 610, 270
        self.box(X, Y, W, H, "Ein Lernschritt, in 4 Teilen")
        steps = ["1 Vorwärts: Netz rät eine Aktion für 128 Beispiele",
                 "2 Fehler:    Loss = wie weit lag es vom Lehrer daneben",
                 "3 Rückwärts: Gradient ∂L/∂w = wie stark ändert dieses",
                 "             Gewicht den Fehler? (Kettenregel, hinten→vorn)",
                 "4 Anpassen:  w ← w − η · ∂L/∂w   (+ Momentum)"]
        for i, s in enumerate(steps):
            self.text(s, (X + 14, Y + 38 + i * 19), self.fs, TXT if i != 2 else TXT)
        k = int(np.argmax(np.abs(self.t.last_grad)))
        g, w, d = self.t.last_grad[k], self.t.params[k], self.t.last_step[k]
        self.text("Konkretes Beispiel: das Gewicht mit dem größten Gradienten", (X + 14, Y + 142), self.f, YEL)
        self.text(param_name(k), (X + 14, Y + 166), self.f)
        self.text(f"aktueller Wert w       = {w:+.4f}", (X + 14, Y + 190), self.f)
        self.text(f"Gradient ∂L/∂w         = {g:+.4f}   "
                  f"({'Fehler wächst mit w → w sinkt' if g > 0 else 'Fehler sinkt mit w → w steigt'})",
                  (X + 14, Y + 210), self.fs, DIM)
        self.text(f"Änderung (η={self.t.lr:.2f}, mit Momentum) = {d:+.5f}", (X + 14, Y + 232), self.f)

    def draw_curves(self):
        X, Y, W, H = 1070, 6, 350, 330
        self.box(X, Y, W, H, "Lernkurve")
        gx, gy, gw, gh = X + 14, Y + 40, W - 28, H - 100
        pygame.draw.rect(self.screen, BG, (gx, gy, gw, gh))

        def line(series, color, top):
            if len(series) < 2:
                return
            stride = max(1, len(series) // 300)
            s = series[::stride]
            pts = [(gx + i / max(1, len(s) - 1) * gw, gy + gh - min(1.0, v / top) * gh) for i, v in enumerate(s)]
            pygame.draw.lines(self.screen, color, False, pts, 2)

        if self.loss_hist:
            line(self.loss_hist, (255, 150, 70), max(self.loss_hist[0], 1e-6))
        n = max(self.t.steps, 1)
        if len(self.acc_hist) > 1:
            pts = [(gx + s / n * gw, gy + gh - a * gh) for s, a in self.acc_hist]
            pygame.draw.lines(self.screen, POS, False, pts, 2)
        self.text("orange: Fehler (Loss, sinkt)", (gx, gy + gh + 6), self.fs, (255, 150, 70))
        self.text("blau: Trefferquote (steigt, 0–100 %)", (gx, gy + gh + 22), self.fs, POS)
        self.text(f"Schritte: 0 … {self.t.steps}", (gx, gy + gh + 38), self.fs, DIM)

    def draw_help(self):
        X, Y, W, H = 1070, 350, 350, 496
        self.box(X, Y, W, H, "Vergleich & Erklärung")
        avg = f"{np.mean(self.scores):.1f}" if self.scores else "–"
        evo = f"{self.evo_score:.1f}" if self.evo_score is not None else "?"
        rows = [("Ø Futter pro Spiel", YEL),
                ("Zufall                0.1", DIM),
                (f"Lehrer (Regeln)    {self.expert_score:6.1f}", TXT),
                (f"Schüler (jetzt)    {avg:>6}", POS),
                (f"Evolution (12 N.)  {evo:>6}", TXT), ("", TXT),
                ("Was ist anders als Evolution?", YEL),
                ("Kein Zufall, keine Generationen:", TXT),
                ("Ein Netz, jeder Schritt kennt die", TXT),
                ("Richtung der Verbesserung.", TXT), ("", TXT),
                ("Aber: Ein Schüler kann nie besser", YEL),
                ("werden als sein Lehrer, denn der", TXT),
                ("Fehler misst nur die Abweichung", TXT),
                ("vom Lehrer, nicht den Score.", TXT), ("", TXT),
                ("1/2/3 Tempo   R Neustart   ESC Ende", DIM)]
        for i, (s, col) in enumerate(rows):
            self.text(s, (X + 14, Y + 40 + i * 22), self.f, col)

    # ---------------------------------------------------------- Hauptschleife
    def run(self, max_frames=None, shot=None):
        frame, running = 0, True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_UP:
                        self.t.lr = min(2.0, self.t.lr * 1.5)
                    elif e.key == pygame.K_DOWN:
                        self.t.lr = max(0.001, self.t.lr / 1.5)
                    elif e.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif e.key == pygame.K_r:
                        self.t.reset()
                        self.loss_hist, self.acc_hist, self.ema = [], [], None
                    self.k = {pygame.K_1: 1, pygame.K_2: 4, pygame.K_3: 25}.get(e.key, self.k)
            self.train_frame()
            self.game_frame()
            self.screen.fill(BG)
            self.draw_game()
            self.draw_network()
            self.draw_step()
            self.draw_curves()
            self.draw_help()
            pygame.display.flip()
            self.clock.tick(1000 if max_frames else 30)
            frame += 1
            if max_frames and frame >= max_frames:
                if shot:
                    pygame.image.save(self.screen, shot)
                running = False
        pygame.quit()


if __name__ == "__main__":
    if "--smoke" in sys.argv:  # Test ohne Benutzer: n Frames, dann Screenshot
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        Live().run(max_frames=int(sys.argv[sys.argv.index("--smoke") + 1]),
                   shot=sys.argv[sys.argv.index("--smoke") + 2])
    else:
        Live().run()
