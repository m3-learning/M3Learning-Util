import matplotlib.pyplot as plt

def remove_all_ticks(*args, **kwargs):
    """
    Remove all x and y ticks from all axes in the current figure.

    This function iterates over all axes in the current figure and sets
    both the x and y ticks to be empty, effectively removing them from
    the plot.
    """
    if len(args) > 0:
        for ax in args:
            plt.setp(ax, xticks=[], yticks=[])
    else:
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
