"""Snake-Umgebung ohne Grafik (Simulation getrennt von Darstellung).

Schnittstelle wie bei Gym: reset() -> state, step(action) -> state, reward, done.
Aktionen sind relativ zur Fahrtrichtung: 0 = geradeaus, 1 = rechts, 2 = links.
"""
import random

import numpy as np

# Richtungen im Uhrzeigersinn: oben, rechts, unten, links als (dx, dy)
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

STRAIGHT, RIGHT, LEFT = 0, 1, 2
N_ACTIONS = 3
N_FEATURES = 11


class SnakeEnv:
    def __init__(self, size=20, seed=None):
        self.size = size
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        mid = self.size // 2
        self.dir = 1  # startet nach rechts
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]  # Kopf zuerst
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self._place_food()
        return self.state()

    def _place_food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.snake]
        self.food = self.rng.choice(free)

    def _is_danger(self, pos):
        x, y = pos
        return not (0 <= x < self.size and 0 <= y < self.size) or pos in self.snake[:-1]

    def _ahead(self, direction):
        dx, dy = DIRS[direction]
        hx, hy = self.snake[0]
        return (hx + dx, hy + dy)

    def state(self):
        """11 Merkmale: Gefahr (geradeaus/rechts/links), Richtung (4), Futterlage (4)."""
        d = self.dir
        danger_straight = self._is_danger(self._ahead(d))
        danger_right = self._is_danger(self._ahead((d + 1) % 4))
        danger_left = self._is_danger(self._ahead((d - 1) % 4))
        hx, hy = self.snake[0]
        fx, fy = self.food
        return np.array([
            danger_straight, danger_right, danger_left,
            d == 0, d == 1, d == 2, d == 3,
            fx < hx, fx > hx, fy < hy, fy > hy,  # Futter links/rechts/oben/unten
        ], dtype=np.float32)

    def step(self, action):
        # relative Aktion in neue absolute Richtung umrechnen
        if action == RIGHT:
            self.dir = (self.dir + 1) % 4
        elif action == LEFT:
            self.dir = (self.dir - 1) % 4

        new_head = self._ahead(self.dir)
        self.steps += 1
        self.steps_since_food += 1

        # Zeitlimit gegen Endlosschleifen: wächst mit der Länge der Schlange
        starved = self.steps_since_food > 100 * len(self.snake)
        if self._is_danger(new_head) or starved:
            return self.state(), -10.0, True

        self.snake.insert(0, new_head)
        reward = 0.0
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward = 10.0
            if len(self.snake) == self.size * self.size:
                return self.state(), reward, True  # Brett voll = gewonnen
            self._place_food()
        else:
            self.snake.pop()
        return self.state(), reward, False


if __name__ == "__main__":
    # Schnelltest: zufällig spielen, Durchschnittsscore zeigen
    env = SnakeEnv(seed=0)
    scores = []
    for _ in range(200):
        env.reset()
        done = False
        while not done:
            _, _, done = env.step(random.randrange(N_ACTIONS))
        scores.append(env.score)
    print(f"Zufall: Ø Score {sum(scores) / len(scores):.2f}, Max {max(scores)}")
