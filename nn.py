"""Kleines neuronales Netz (MLP) nur mit NumPy, nur Forward-Pass.

Alle Parameter liegen in EINEM flachen Vektor (das "Genom"), damit Evolution
sie einfach kreuzen und mutieren kann. Layout (für n_hid = 12, 183 Zahlen):
    [ W1 (11 x n_hid) | b1 (n_hid) | W2 (n_hid x 3) | b2 (3) ]
"""
import numpy as np

N_IN, N_HID, N_OUT = 11, 12, 3


def layout(n_hid):
    """Start-Indizes der Teile im flachen Vektor: (W1, b1, W2, b2, gesamt)."""
    w1 = 0
    b1 = w1 + N_IN * n_hid
    w2 = b1 + n_hid
    b2 = w2 + n_hid * N_OUT
    return w1, b1, w2, b2, b2 + N_OUT


def n_params(n_hid=N_HID):
    return layout(n_hid)[4]


# Standardgröße, wird vom Live-Fenster (evolve.py) genutzt
W1_AT, B1_AT, W2_AT, B2_AT, N_PARAMS = layout(N_HID)


class MLP:
    def __init__(self, params, n_hid=N_HID):
        w1, b1, w2, b2, end = layout(n_hid)
        self.params = params
        self.W1 = params[w1:b1].reshape(N_IN, n_hid)
        self.b1 = params[b1:w2]
        self.W2 = params[w2:b2].reshape(n_hid, N_OUT)
        self.b2 = params[b2:end]

    def forward(self, x):
        """Gibt (Ausgabe, (z_hidden, a_hidden)) zurück.

        z = Summe(Gewicht * Eingang) + Bias     (gewichtete Summe)
        a = tanh(z)                              (Aktivierungsfunktion)
        """
        z = x @ self.W1 + self.b1
        a = np.tanh(z)
        out = a @ self.W2 + self.b2
        return out, (z, a)

    def act(self, x):
        out, _ = self.forward(x)
        return int(np.argmax(out))


def random_params(rng, n=None, n_hid=N_HID):
    """Zufällige Startgewichte (Generation 0 = reiner Zufall)."""
    size = n_params(n_hid)
    shape = (size,) if n is None else (n, size)
    return rng.normal(0.0, 0.5, size=shape)
