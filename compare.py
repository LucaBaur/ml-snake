"""Vergleich: Wie viele versteckte Neuronen braucht die Snake-KI?

Ohne Fenster. Pro Größe mehrere unabhängige Evolutionsläufe (Wiederholungen),
weil ein einzelner Lauf zufallsbedingt stark schwankt. Am Ende wird jedes
Netz auf NEUEN Spielen getestet, die es im Training nie gesehen hat.

    py compare.py                          # 4, 8, 12, 32 Neuronen, 5 Läufe, 60 Gen
    py compare.py --sizes 2 6 16 --runs 3 --gens 40
    py compare.py --plot-only              # nur Diagramm aus gespeicherten Daten
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np

from evolve import POP, evaluate, next_generation
from nn import n_params, random_params

SIGMA, RATE = 0.3, 0.10  # feste Mutation, damit nur die Netzgröße variiert
TEST_SEED, TEST_GAMES = 10_000, 50  # Testspiele: andere Seeds als im Training (gen*10)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_one(pool, n_hid, run, gens):
    """Ein kompletter Evolutionslauf. Gibt (Lernkurve, Testscore) zurück."""
    rng = np.random.default_rng(1000 * n_hid + run)  # reproduzierbar
    pop = random_params(rng, POP, n_hid)
    curve = []
    for gen in range(gens):
        res = list(pool.map(evaluate, pop, repeat(gen * 10), repeat(n_hid), chunksize=8))
        fit = np.array([r[0] for r in res])
        order = np.argsort(-fit)
        pop = pop[order]
        curve.append(res[order[0]][1])  # Ø-Score des Siegers dieser Generation
        pop, _ = next_generation(pop, rng, SIGMA, RATE)
    # Nach dem letzten next_generation steht der Sieger wegen Elitismus noch in pop[0]
    test = evaluate(pop[0], TEST_SEED, n_hid, TEST_GAMES)[1]
    return curve, test


def collect(sizes, runs, gens):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "hidden_size.json")
    data = {"gens": gens, "sizes": {}}
    pool = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))
    total, done, t0 = len(sizes) * runs, 0, time.time()
    for n_hid in sizes:
        entry = data["sizes"][str(n_hid)] = {"params": n_params(n_hid), "curves": [], "tests": []}
        for run in range(runs):
            curve, test = run_one(pool, n_hid, run, gens)
            entry["curves"].append(curve)
            entry["tests"].append(test)
            done += 1
            eta = (time.time() - t0) / done * (total - done)
            print(f"[{done}/{total}] {n_hid:>3} Neuronen, Lauf {run + 1}: "
                  f"Test-Score {test:5.1f}   (noch ca. {eta / 60:.1f} min)", flush=True)
            with open(path, "w") as f:  # nach jedem Lauf sichern
                json.dump(data, f)
    pool.shutdown()
    return data


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(data["sizes"], key=int)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(sizes)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})

    for c, s in zip(colors, sizes):
        curves = np.array(data["sizes"][s]["curves"])
        mean, std = curves.mean(0), curves.std(0)
        x = np.arange(len(mean))
        ax1.plot(x, mean, color=c, lw=2, label=f"{s} Neuronen ({data['sizes'][s]['params']} Parameter)")
        ax1.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)
    ax1.set(title="Lernkurve (Mittel ± Streuung über die Läufe)", xlabel="Generation",
            ylabel="Ø Futter pro Spiel (bestes Netz)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    rng = np.random.default_rng(0)
    for i, (c, s) in enumerate(zip(colors, sizes)):
        t = np.array(data["sizes"][s]["tests"])
        ax2.scatter(i + rng.uniform(-0.12, 0.12, len(t)), t, color=c, s=45, alpha=0.8)
        ax2.hlines(t.mean(), i - 0.25, i + 0.25, color="black", lw=2)
    ax2.set_xticks(range(len(sizes)), [f"{s}" for s in sizes])
    ax2.set(title=f"Test auf {TEST_GAMES} neuen Spielen (Punkt = ein Lauf, Strich = Mittel)",
            xlabel="versteckte Neuronen", ylabel="Ø Futter pro Spiel")
    ax2.grid(alpha=0.3)

    fig.suptitle("Wie viele versteckte Neuronen braucht Snake?", fontsize=14)
    fig.tight_layout()
    png = os.path.join(OUT, "hidden_size.png")
    fig.savefig(png, dpi=110)
    print("Diagramm:", png)

    print("\nGröße | Parameter | Test Ø ± Std | bester Lauf")
    for s in sizes:
        t = np.array(data["sizes"][s]["tests"])
        print(f"{s:>5} | {data['sizes'][s]['params']:>9} | {t.mean():5.1f} ± {t.std():4.1f} | {t.max():5.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 12, 32])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        with open(os.path.join(OUT, "hidden_size.json")) as f:
            plot(json.load(f))
    else:
        plot(collect(a.sizes, a.runs, a.gens))
