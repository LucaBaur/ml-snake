"""Live-Fenster: DQN lernt Snake. Gespielt wird das TRAININGS-Spiel selbst, du siehst also
direkt, wann ein Zug Zufall (ε) und wann er eine Entscheidung des Netzes ist.

Tasten:  ↑/↓ Lernrate   1/2/3 Tempo (1 / 4 / 25 Schritte pro Bild)   SPACE Pause   R Neustart   ESC Ende
"""
import os
import sys

import numpy as np
import pygame

from backprop_live import Live
from dqn import Runner
from evolve import BG, DIM, NEG, OUT_NAMES, POS, TXT, YEL
from nn import MLP

ORANGE = (255, 150, 70)


class _View:
    """Schmaler Adapter: gibt dem geerbten draw_network() die Felder, die es erwartet."""

    def __init__(self, owner, params):
        self.owner, self.params, self.last_step, self.lr = owner, params, np.zeros_like(params), 1e-3

    def reset(self):  # wird von Live.run() bei Taste R aufgerufen
        self.owner.restart()


class DqnLive(Live):
    # Bewusst KEIN super().__init__(): das würde Trainingsdaten für Backprop sammeln.
    def __init__(self):
        self._init_window("ML-Snake – DQN")
        self.paused, self.k = False, 1
        self.loss_hist, self.acc_hist, self.ema = [], [], None  # Felder, die Live.run() beim Reset anfasst
        self.restart()

    def restart(self):
        self.r = Runner()
        self.t = _View(self, self.r.flat_params())
        self.inp, self.out, self.acts = self.r.state, None, None
        self.loss_hist, self.acc_hist, self.ema = [], [], None

    # ---------------------------------------------------------- Logik
    def train_frame(self):
        if self.paused:
            return
        for g in self.r.opt.param_groups:
            g["lr"] = self.t.lr
        for _ in range(self.k):
            self.r.step()
        new = self.r.flat_params()
        self.t.last_step, self.t.params = new - self.t.params, new

    def game_frame(self):
        self.inp = self.r.state
        self.out, self.acts = MLP(self.t.params, self.r.hidden).forward(self.inp)

    # ---------------------------------------------------------- Zeichnen
    def draw_game(self):
        r, env, c, x0, y0 = self.r, self.r.env, 20, 10, 50
        self.box(x0 - 4, y0 - 44, 408, 600, "Trainings-Spiel (live)")
        for x in range(20):
            for y in range(20):
                pygame.draw.rect(self.screen, (32, 32, 44), (x0 + x * c, y0 + y * c, c, c), 1)
        head_color = (255, 214, 64) if r.last_random else (140, 255, 170)  # gelb = Zufallszug
        for i, (x, y) in enumerate(env.snake):
            pygame.draw.rect(self.screen, head_color if i == 0 else (80, 200, 120),
                             (x0 + x * c + 1, y0 + y * c + 1, c - 2, c - 2), border_radius=4)
        fx, fy = env.food
        pygame.draw.circle(self.screen, (240, 80, 80), (x0 + fx * c + c // 2, y0 + fy * c + c // 2), c // 2 - 3)

        y = y0 + 20 * c + 10
        eps = r.eps
        pygame.draw.rect(self.screen, BG, (x0, y + 22, 300, 12))
        pygame.draw.rect(self.screen, YEL, (x0, y + 22, int(300 * eps), 12))
        self.text(f"ε = {eps:.2f}: {eps:.0%} der Züge sind Zufall", (x0, y), self.f, YEL)
        self.text("Letzter Zug: ZUFALL" if r.last_random else "Letzter Zug: vom Netz gewählt",
                  (x0, y + 40), self.f, YEL if r.last_random else POS)
        avg = f"{np.mean(r.ep_scores[-20:]):.1f}" if r.ep_scores else "–"
        test = f"{r.eval_hist[-1][1]:.1f}" if r.eval_hist else "–"
        lines = [f"Score jetzt: {env.score}   Ø letzte 20 Trainingsspiele: {avg}",
                 f"Test-Score (nur Netz, ε=0): {test}",
                 f"Schritt {r.steps}   Spiel {r.episodes}   Buffer {r.n}/{r.cap}",
                 f"Loss: {r.last_loss:.3f}" if r.last_loss is not None else "Lernen startet nach 1000 Schritten",
                 f"Lernrate = {self.t.lr:.4f} (↑/↓)   γ = {r.gamma}   Reward-Shaping {r.shape}"]
        for i, s in enumerate(lines):
            self.text(s, (x0, y + 66 + i * 20), self.f if i < 3 else self.fs, TXT if i < 3 else DIM)
        if self.paused:
            self.text("PAUSE (SPACE)", (x0 + 300, y + 66), self.f, YEL)

    def draw_step(self):
        X, Y, W, H = 440, 576, 610, 270
        self.box(X, Y, W, H, "Ein DQN-Schritt, in 5 Teilen   (Ausgänge = Q-Werte)")
        steps = ["1 Wählen:  mit Wahrsch. ε Zufall, sonst Aktion mit größtem Q",
                 "2 Handeln: Belohnung r + neuer Zustand s' in den Replay Buffer",
                 "3 Ziehen:  64 zufällige ALTE Erfahrungen aus dem Buffer",
                 "4 Ziel:    y = r + γ · max Q_ziel(s')   (eingefrorene Kopie)",
                 "5 Anpassen: Fehler (Q(s,a) − y)² → Gradient → Adam-Schritt"]
        for i, s in enumerate(steps):
            self.text(s, (X + 14, Y + 38 + i * 19), self.fs)
        ex = self.r.last_example
        self.text("Beispiel aus dem letzten Lernschritt (1 von 64):", (X + 14, Y + 142), self.f, YEL)
        if ex:
            self.text(f"Aktion {OUT_NAMES[ex['a']]}, Belohnung r = {ex['r']:+.2f}"
                      + ("  (Spiel vorbei)" if ex["done"] else ""), (X + 14, Y + 166), self.f)
            self.text(f"Netz sagt:   Q(s,a)                 = {ex['q']:+.2f}", (X + 14, Y + 190), self.f)
            self.text(f"Ziel:        r + γ·max Q_ziel(s')   = {ex['y']:+.2f}", (X + 14, Y + 210), self.f)
            self.text(f"Abweichung {ex['q'] - ex['y']:+.2f} → Gewichte werden so verschoben, dass Q näher an y liegt",
                      (X + 14, Y + 234), self.fs, DIM)
        nxt = self.r.sync_every - self.r.steps % self.r.sync_every
        self.text(f"Ziel-Netz wird in {nxt} Schritten aktualisiert", (X + 14, Y + 252), self.fs, DIM)

    def draw_curves(self):
        X, Y, W, H = 1070, 6, 350, 330
        self.box(X, Y, W, H, "Lernkurve")
        r = self.r
        gx, gy, gw, gh = X + 14, Y + 40, W - 28, H - 106
        pygame.draw.rect(self.screen, BG, (gx, gy, gw, gh))
        n = max(r.steps, 1)
        avg = [float(np.mean(r.ep_scores[max(0, i - 19):i + 1])) for i in range(len(r.ep_scores))]
        top = max([10.0] + avg + [s for _, s in r.eval_hist]) * 1.1

        def px(step, v):
            return (gx + step / n * gw, gy + gh - min(1.0, v / top) * gh)

        pygame.draw.lines(self.screen, (110, 90, 40), False,
                          [px(s, top * max(r.eps_end, 1 - (1 - r.eps_end) * s / r.eps_decay))
                           for s in np.linspace(0, n, 60)], 1)
        if len(avg) > 1:
            stride = max(1, len(avg) // 300)
            pygame.draw.lines(self.screen, (140, 255, 170), False,
                              [px(s, v) for s, v in list(zip(r.ep_steps, avg))[::stride]], 2)
        if len(r.eval_hist) > 1:
            pygame.draw.lines(self.screen, POS, False, [px(s, v) for s, v in r.eval_hist], 2)
        for s, v in r.eval_hist:
            pygame.draw.circle(self.screen, POS, px(s, v), 4)
        self.text(f"{top:.0f}", (X + 2, gy - 4), self.fs, DIM)
        self.text("blau: Test-Score, nur Netz (ε=0)", (gx, gy + gh + 6), self.fs, POS)
        self.text("grün: Ø Trainingsspiele (mit Zufall)", (gx, gy + gh + 22), self.fs, (140, 255, 170))
        self.text("gelbbraun: ε (Zufallsanteil, sinkt)", (gx, gy + gh + 38), self.fs, (170, 140, 60))
        self.text(f"Schritte: 0 … {r.steps}", (gx, gy + gh + 54), self.fs, DIM)

    def draw_help(self):
        X, Y, W, H = 1070, 350, 350, 496
        self.box(X, Y, W, H, "Vergleich & Erklärung")
        test = f"{self.r.eval_hist[-1][1]:.1f}" if self.r.eval_hist else "–"
        rows = [("Ø Futter pro Spiel (Test)", YEL),
                ("Zufall                0.1", DIM),
                ("Lehrer / Backprop-Schüler 28.3", TXT),
                ("Evolution (12 N.)    65.4", TXT),
                (f"DQN (jetzt)         {test:>6}", POS), ("", TXT),
                ("Replay Buffer", YEL), ("Gelernt wird aus zufälligen alten", TXT),
                ("Erfahrungen, nicht der letzten.", DIM), ("Das bricht die Abhängigkeit aufein-", DIM),
                ("anderfolgender Schritte.", DIM), ("", TXT),
                ("Reward Shaping", YEL), ("Ohne Zwischenbelohnung 'näher am", TXT),
                ("Futter' lernte DQN fast nichts", TXT), ("(Test-Score 0.3 statt ~25).", DIM), ("", TXT),
                ("1/2/3 Tempo  SPACE Pause  R Neu", DIM)]
        for i, (s, col) in enumerate(rows):
            self.text(s, (X + 14, Y + 40 + i * 22), self.f, col)


if __name__ == "__main__":
    if "--smoke" in sys.argv:  # Test ohne Benutzer: Frames laufen lassen, Screenshot speichern
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        i = sys.argv.index("--smoke")
        win = DqnLive()
        win.k = 25
        win.run(max_frames=int(sys.argv[i + 1]), shot=sys.argv[i + 2])
    else:
        DqnLive().run()
