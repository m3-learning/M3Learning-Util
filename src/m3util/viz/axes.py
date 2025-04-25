import matplotlib.pyplot as plt

def remove_all_ticks():
    """
    Remove all x and y ticks from all axes in the current figure.

    This function iterates over all axes in the current figure and sets
    both the x and y ticks to be empty, effectively removing them from
    the plot.
    """
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
