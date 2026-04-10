import matplotlib.pyplot as plt


def plot() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.set_aspect("equal")
    ax.set_xlim(left=0.0, right=None)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return fig, ax
