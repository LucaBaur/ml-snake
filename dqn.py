"""Deep Q-Learning (DQN) mit PyTorch für Snake.

Gleiches Netz wie bei Evolution und Imitation (11 -> N_HID -> 3, tanh), gleiche
Umgebung, gleiche Zustände. Anders ist nur WIE gelernt wird: Der Score kommt als
Belohnung aus der Umgebung, die Anpassung läuft über Gradienten.

    py dqn.py [--steps 60000] [--hidden 12]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import N_ACTIONS, N_FEATURES, SnakeEnv
from imitation import BOARD, OUT, play
from nn import MLP, N_HID

torch.set_num_threads(1)  # winziges Netz: Mehrfach-Threads kosten mehr, als sie bringen


class Runner:
    """Ein DQN-Agent samt Umgebung. step() = ein Umgebungsschritt + ein Lernschritt."""

    def __init__(self, hidden=N_HID, seed=0, gamma=0.95, lr=1e-3, batch=64, capacity=50_000,
                 warmup=1_000, eps_end=0.02, eps_decay=15_000, sync_every=500, eval_every=5_000,
                 shape=0.5):
        self.hidden, self.gamma, self.batch, self.shape = hidden, gamma, batch, shape
        self.warmup, self.eps_end, self.eps_decay = warmup, eps_end, eps_decay
        self.sync_every, self.eval_every = sync_every, eval_every
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        def make():
            return nn.Sequential(nn.Linear(N_FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, N_ACTIONS))

        self.q, self.target = make(), make()
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        # Replay Buffer als Ringspeicher: (s, a, r, s', done)
        self.cap, self.n, self.ptr = capacity, 0, 0
        self.S = np.zeros((capacity, N_FEATURES), np.float32)
        self.S2 = np.zeros((capacity, N_FEATURES), np.float32)
        self.A = np.zeros(capacity, np.int64)
        self.R = np.zeros(capacity, np.float32)
        self.D = np.zeros(capacity, np.float32)

        self.env = SnakeEnv(BOARD, seed=seed)
        self.state = self.env.reset()
        self.steps, self.episodes = 0, 0
        self.ep_scores, self.ep_steps, self.eval_hist, self.loss_hist = [], [], [], []
        self.last_random, self.last_example, self.last_loss = False, None, None

    # -- Exploration: ε sinkt linear von 1.0 (nur Zufall) auf eps_end
    @property
    def eps(self):
        return max(self.eps_end, 1.0 - (1.0 - self.eps_end) * self.steps / self.eps_decay)

    def q_values(self, state):
        with torch.no_grad():
            return self.q(torch.from_numpy(state)).numpy()

    def remember(self, s, a, r, s2, done):
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i], self.D[i] = s, a, r, s2, float(done)
        self.ptr, self.n = (i + 1) % self.cap, min(self.n + 1, self.cap)

    def learn(self):
        idx = self.rng.integers(0, self.n, self.batch)  # zufällige alte Erfahrungen
        S, S2 = torch.from_numpy(self.S[idx]), torch.from_numpy(self.S2[idx])
        A, R, D = torch.from_numpy(self.A[idx]), torch.from_numpy(self.R[idx]), torch.from_numpy(self.D[idx])
        q_sa = self.q(S).gather(1, A[:, None]).squeeze(1)       # Q(s, a) der echt gewählten Aktion
        with torch.no_grad():                                    # Ziel: kein Gradient durch das Ziel
            y = R + self.gamma * (1.0 - D) * self.target(S2).max(dim=1).values
        loss = F.smooth_l1_loss(q_sa, y)                         # Huber: robust gegen Ausreißer
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.last_loss = float(loss.detach())
        self.last_example = dict(a=int(A[0]), r=float(R[0]), done=bool(D[0]),
                                 q=float(q_sa[0].detach()), y=float(y[0]))

    def potential(self):
        """Φ(s): je näher der Kopf am Futter, desto höher (immer >= 0, max = shape * 38)."""
        (hx, hy), (fx, fy) = self.env.snake[0], self.env.food
        return self.shape * (2 * BOARD - 2 - (abs(hx - fx) + abs(hy - fy)))

    def step(self):
        s = self.state
        self.last_random = bool(self.rng.random() < self.eps)
        a = int(self.rng.integers(N_ACTIONS)) if self.last_random else int(np.argmax(self.q_values(s)))
        phi = self.potential()
        s2, r, done = self.env.step(a)
        # Potentialbasiertes Reward Shaping: r' = r + γ·Φ(s') − Φ(s); Endzustand hat Φ = 0
        r += self.gamma * (0.0 if done else self.potential()) - phi
        self.remember(s, a, r, s2, done)
        self.steps += 1
        if done:
            self.ep_scores.append(self.env.score)
            self.ep_steps.append(self.steps)
            self.episodes += 1
            self.state = self.env.reset()
        else:
            self.state = s2
        if self.n >= self.warmup:
            self.learn()
            self.loss_hist.append(self.last_loss)
        if self.steps % self.sync_every == 0:
            self.target.load_state_dict(self.q.state_dict())    # Target-Netz nachziehen
        if self.steps % self.eval_every == 0:
            self.eval_hist.append((self.steps, self.evaluate()))

    def flat_params(self):
        """Gewichte im flachen Layout von nn.py (für Darstellung und numpy-Auswertung)."""
        l1, l2 = self.q[0], self.q[2]
        parts = [l1.weight.detach().numpy().T.ravel(), l1.bias.detach().numpy(),
                 l2.weight.detach().numpy().T.ravel(), l2.bias.detach().numpy()]
        return np.concatenate(parts).astype(np.float64)

    def evaluate(self, games=50):
        """Greedy (ε = 0, kein Lernen) auf den gleichen Testspielen wie bei den anderen Verfahren."""
        net = MLP(self.flat_params(), self.hidden)
        return play(net.act, games)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--hidden", type=int, default=N_HID)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape", type=float, default=0.5, help="Stärke des Reward Shaping (0 = aus)")
    a = ap.parse_args()
    r = Runner(hidden=a.hidden, seed=a.seed, shape=a.shape)
    print(f"{'Schritt':>8} {'ε':>5} {'Ø Train-Score':>14} {'Test-Score':>11} {'Loss':>7}")
    while r.steps < a.steps:
        r.step()
        if r.steps % r.eval_every == 0:
            recent = np.mean(r.ep_scores[-50:]) if r.ep_scores else 0.0
            print(f"{r.steps:>8} {r.eps:>5.2f} {recent:>14.2f} {r.eval_hist[-1][1]:>11.1f} "
                  f"{np.mean(r.loss_hist[-500:]):>7.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "dqn.json"), "w") as f:
        json.dump({"hidden": a.hidden, "eval": r.eval_hist}, f)
