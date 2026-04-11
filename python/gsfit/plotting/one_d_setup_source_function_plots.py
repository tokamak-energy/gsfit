import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot() -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, gridspec_kw={"wspace": 0.1})

    ax[0].set_xlim(0.0, 1.0)
    ax[1].set_xlim(0.0, 1.0)
    ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax[0].grid()
    ax[1].grid()

    # ax[1].yaxis.tick_right()
    ax[0].set_xlabel("psi_n [-]")
    ax[1].set_xlabel("psi_n [-]")
    ax[0].set_ylabel("p_prime")
    ax[1].set_ylabel("ff_prime")

    # Plot zero line
    ax[0].plot([0.0, 1.0], [0.0, 0.0], color="black", linestyle="-", linewidth=1.0)
    ax[1].plot([0.0, 1.0], [0.0, 0.0], color="black", linestyle="-", linewidth=1.0)  #

    # Auto-refresh the legend on ax[1] whenever the figure is redrawn,
    # so that any new labeled artists added after setup appear automatically.
    def _auto_legend(event: object, _updating: list[bool] = [False]) -> None:
        # Guard against infinite recursion: legend() triggers a draw,
        # which would re-enter this callback without the guard.
        if _updating[0]:
            return
        handles, labels = ax[1].get_legend_handles_labels()
        if labels:
            _updating[0] = True
            ax[1].legend()
            fig.canvas.draw_idle()
            _updating[0] = False

    fig.canvas.mpl_connect("draw_event", _auto_legend)

    return fig, ax
