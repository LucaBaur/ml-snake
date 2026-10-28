"""pygame-Darstellung: spielt eine Policy (Funktion state -> action) sichtbar ab."""
import random

import pygame

from env import N_ACTIONS, SnakeEnv

CELL = 28
BG, GRID, SNAKE, HEAD, FOOD, TEXT = (18, 18, 24), (30, 30, 40), (80, 200, 120), (140, 255, 170), (240, 80, 80), (220, 220, 230)


def watch(policy, size=20, fps=12, episodes=None, title="ML-Snake"):
    """Zeigt Spiele, bis das Fenster geschlossen wird (oder `episodes` Spiele vorbei sind)."""
    pygame.init()
    screen = pygame.display.set_mode((size * CELL, size * CELL + 32))
    pygame.display.set_caption(title)
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()
    env = SnakeEnv(size)
    played = 0
    state, running = env.reset(), True

    while running and (episodes is None or played < episodes):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state, _, done = env.step(policy(state))

        screen.fill(BG)
        for x in range(size):
            for y in range(size):
                pygame.draw.rect(screen, GRID, (x * CELL, y * CELL, CELL, CELL), 1)
        for i, (x, y) in enumerate(env.snake):
            color = HEAD if i == 0 else SNAKE
            pygame.draw.rect(screen, color, (x * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4), border_radius=6)
        fx, fy = env.food
        pygame.draw.circle(screen, FOOD, (fx * CELL + CELL // 2, fy * CELL + CELL // 2), CELL // 2 - 4)
        label = font.render(f"Score {env.score}   Spiel {played + 1}", True, TEXT)
        screen.blit(label, (8, size * CELL + 6))
        pygame.display.flip()
        clock.tick(fps)

        if done:
            played += 1
            state = env.reset()
    pygame.quit()


if __name__ == "__main__":
    # Ohne Lernen: Zufallspolicy, damit man die Umgebung einmal sieht
    watch(lambda s: random.randrange(N_ACTIONS), episodes=5)
