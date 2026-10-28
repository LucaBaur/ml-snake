"""Imitation Learning mit selbstgebauter Backpropagation (nur NumPy).

Ein regelbasierter "Lehrer" spielt Snake. Das Netz bekommt (Zustand, Lehrer-Aktion)
als Trainingsdaten und lernt per Gradientenabstieg, den Lehrer nachzuahmen.
Das ist Supervised Learning: Klassifikation in 3 Klassen (geradeaus/rechts/links).

    py imitation.py        # Gradienten-Check, Daten sammeln, trainieren, vergleichen
"""
import os

import numpy as np

from env import LEFT, RIGHT, STRAIGHT, SnakeEnv
from nn import MLP, N_HID, layout, random_params

BOARD = 20
TEST_SEED = 10_000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ------------------------------------------------------------------ Lehrer
def expert(s):
    """Regel-KI, die NUR die 11 Merkmale nutzt (also exakt erlernbar ist).

    Regel: nie in eine direkte Gefahr laufen; unter den sicheren Zügen den wählen,
    der zum Futter hin zeigt; sonst irgendeinen sicheren; bei Gleichstand geradeaus.
    """
    d = int(np.argmax(s[3:7]))  # aktuelle Richtung: 0 oben, 1 rechts, 2 unten, 3 links
    food_in_dir = {0: s[9], 1: s[8], 2: s[10], 3: s[7]}  # liegt Futter in dieser Richtung?
    best, best_score = STRAIGHT, -1
    for action, new_dir in ((STRAIGHT, d), (RIGHT, (d + 1) % 4), (LEFT, (d - 1) % 4)):
        danger = s[action]  # Merkmale 0..2 sind Gefahr geradeaus/rechts/links
        score = 0 if danger else (2 if food_in_dir[new_dir] else 1)
        if score > best_score:
            best, best_score = action, score
    return best


def play(policy, games, seed0=TEST_SEED):
    """Mittlerer Score einer Policy (Funktion state -> action) über mehrere Spiele."""
    scores = []
    for g in range(games):
        env = SnakeEnv(BOARD, seed=seed0 + g)
        state, done = env.state(), False
        while not done:
            state, _, done = env.step(policy(state))
        scores.append(env.score)
    return float(np.mean(scores))


def collect(n_games=300, eps=0.1, seed=0):
    """Sammelt (Zustand, Lehrer-Aktion). Mit Wahrscheinlichkeit eps wird eine ZUFÄLLIGE
    Aktion ausgeführt, das Label bleibt aber die Lehrer-Aktion. So sieht das Netz auch
    Lagen, in die der perfekte Lehrer nie käme (Gegenmittel zum *Distribution Shift*)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for g in range(n_games):
        env = SnakeEnv(BOARD, seed=seed * 100_000 + g)
        state, done = env.state(), False
        while not done:
            label = expert(state)
            X.append(state)
            y.append(label)
            action = int(rng.integers(3)) if rng.random() < eps else label
            state, _, done = env.step(action)
    return np.array(X, dtype=np.float64), np.array(y)


# ------------------------------------------------------- Backpropagation
def softmax(out):
    e = np.exp(out - out.max(axis=1, keepdims=True))  # Max abziehen: numerisch stabil
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grads(net, X, y):
    """Cross-Entropy-Fehler und Gradient aller Parameter (als flacher Vektor).

    Vorwärts:  z = X·W1 + b1,  a = tanh(z),  out = a·W2 + b2,  p = softmax(out)
    Fehler:    L = -Mittel( log p[richtige Klasse] )
    Rückwärts: Kettenregel Schicht für Schicht, von hinten nach vorn.
    """
    B = len(X)
    out, (z, a) = net.forward(X)
    p = softmax(out)
    loss = -np.mean(np.log(p[np.arange(B), y] + 1e-12))

    d_out = p.copy()
    d_out[np.arange(B), y] -= 1.0       # ∂L/∂out = p - one_hot(y)   (Softmax+CE ergibt das)
    d_out /= B                          # Mittel über den Batch
    gW2 = a.T @ d_out                   # ∂L/∂W2
    gb2 = d_out.sum(axis=0)
    d_a = d_out @ net.W2.T              # Fehler zurück zur versteckten Schicht
    d_z = d_a * (1.0 - a ** 2)          # tanh'(z) = 1 - tanh(z)²
    gW1 = X.T @ d_z                     # ∂L/∂W1
    gb1 = d_z.sum(axis=0)
    return loss, np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])


def accuracy(net, X, y):
    return float((np.argmax(net.forward(X)[0], axis=1) == y).mean())


def gradient_check(n_checks=15, seed=0):
    """Beweis, dass die Backprop stimmt: analytischer Gradient vs. numerisches
    Ausprobieren (Parameter kurz um ±h verschieben und die Fehleränderung messen)."""
    rng = np.random.default_rng(seed)
    X, y = collect(n_games=3, seed=seed)
    X, y = X[:64], y[:64]
    params = random_params(rng)
    net = MLP(params)
    _, grad = loss_and_grads(net, X, y)
    h, worst = 1e-5, 0.0
    for i in rng.choice(len(params), n_checks, replace=False):
        old = params[i]
        params[i] = old + h
        lp, _ = loss_and_grads(net, X, y)
        params[i] = old - h
        lm, _ = loss_and_grads(net, X, y)
        params[i] = old
        numeric = (lp - lm) / (2 * h)
        worst = max(worst, abs(grad[i] - numeric) / max(1e-8, abs(grad[i]) + abs(numeric)))
    return worst


class Trainer:
    """Mini-Batch-SGD mit Momentum. Hält Gewichte, Geschwindigkeit und Datenreihenfolge."""

    def __init__(self, X, y, lr=0.1, momentum=0.9, batch=128, seed=0):
        self.rng = np.random.default_rng(seed)
        n_val = len(X) // 5
        perm = self.rng.permutation(len(X))
        self.Xv, self.yv = X[perm[:n_val]], y[perm[:n_val]]          # 20 % zum Prüfen
        self.X, self.y = X[perm[n_val:]], y[perm[n_val:]]            # 80 % zum Lernen
        self.lr, self.momentum, self.batch = lr, momentum, batch
        self.reset()

    def reset(self):
        self.params = random_params(self.rng)
        self.net = MLP(self.params)
        self.vel = np.zeros_like(self.params)
        self.last_step = np.zeros_like(self.params)  # letzte tatsächliche Änderung (zum Anzeigen)
        self.last_grad = np.zeros_like(self.params)
        self.order, self.pos, self.epoch, self.steps = self.rng.permutation(len(self.X)), 0, 0, 0

    def step(self):
        """Ein Lernschritt auf einem Mini-Batch. Gibt den Fehler dieses Batches zurück."""
        if self.pos + self.batch > len(self.order):
            self.order, self.pos, self.epoch = self.rng.permutation(len(self.X)), 0, self.epoch + 1
        idx = self.order[self.pos:self.pos + self.batch]
        self.pos += self.batch
        loss, grad = loss_and_grads(self.net, self.X[idx], self.y[idx])
        self.vel = self.momentum * self.vel - self.lr * grad   # Momentum: Richtung "merkt" sich Schwung
        self.params += self.vel                                 # w ← w + v   (in place, Netz sieht es sofort)
        self.last_step, self.last_grad = self.vel.copy(), grad
        self.steps += 1
        return loss

    def val_accuracy(self):
        return accuracy(self.net, self.Xv, self.yv)


if __name__ == "__main__":
    print(f"Gradienten-Check: größter relativer Fehler {gradient_check():.1e}  (< 1e-6 = Backprop stimmt)")
    X, y = collect(n_games=300)
    print(f"{len(X)} Trainingsbeispiele, davon geradeaus {np.mean(y == 0):.0%}, "
          f"rechts {np.mean(y == 1):.0%}, links {np.mean(y == 2):.0%}, "
          f"verschiedene Zustände: {len(np.unique(X, axis=0))}")
    t = Trainer(X, y)
    for ep in range(1, 31):
        while t.epoch < ep:
            t.step()
        if ep % 5 == 0:
            print(f"Epoche {ep:>2}: Genauigkeit (ungesehene Daten) {t.val_accuracy():.1%}")
    print("Zufall:   Ø", play(lambda s: int(np.random.randint(3)), 50))
    print("Lehrer:   Ø", play(expert, 50))
    print("Schüler:  Ø", play(t.net.act, 50))
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "imitation.npy"), t.params)
