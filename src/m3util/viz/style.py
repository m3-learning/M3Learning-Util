import seaborn as sns
import matplotlib.pyplot as plt



def set_style(name="default"):
    """Function to implement custom default style for graphs

    Args:
        name (str, optional): style name. Defaults to "default".
    """
    if name == "default":
        try:
            # resetting default seaborn style
            sns.reset_orig()

            print(f"{name} set for seaborn")

        except Exception:
            pass

        try:
            # setting default plotting params
            plt.rcParams["image.cmap"] = "magma"
            plt.rcParams["axes.labelsize"] = 18
            plt.rcParams["xtick.labelsize"] = 16
            plt.rcParams["ytick.labelsize"] = 16
            plt.rcParams["figure.titlesize"] = 20
            plt.rcParams["xtick.direction"] = "in"
            plt.rcParams["ytick.direction"] = "in"
            plt.rcParams["xtick.top"] = True
            plt.rcParams["ytick.right"] = True
            print(f"{name} set for matplotlib")
        except Exception:
            pass

    if name == "printing":
        try:
            # resetting default seaborn style
            sns.reset_orig()

            print(f"{name} set for seaborn")

        except Exception:
            pass

        # setting default plotting params
        plt.rcParams["image.cmap"] = "viridis"
        plt.rcParams["axes.labelsize"] = 8
        plt.rcParams["xtick.labelsize"] = 6
        plt.rcParams["ytick.labelsize"] = 6
        plt.rcParams["figure.titlesize"] = 8
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.right"] = True
        plt.rcParams["lines.markersize"] = 0.5
        plt.rcParams["axes.grid"] = False
        plt.rcParams["lines.linewidth"] = 0.5
        plt.rcParams["axes.linewidth"] = 0.5
        plt.rcParams["legend.fontsize"] = 5
        plt.rcParams["legend.loc"] = "upper left"
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["font.size"] = 8
