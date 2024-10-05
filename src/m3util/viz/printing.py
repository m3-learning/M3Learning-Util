from m3util.util.IO import make_folder
from m3util.viz.text import labelfigs
from m3util.viz.text import bring_text_to_front


class printer:
    """Class to save figures to a folder"""

    def __init__(
        self, dpi=600, basepath="./", fileformats=["png", "svg"], verbose=True
    ):
        """Initializes the printer class

        Args:
            dpi (int, optional): the resolution of the image. Defaults to 600.
            basepath (str, optional): basepath where files are saved. Defaults to './'.
        """
        self.dpi = dpi
        self.basepath = basepath
        self.fileformats = fileformats
        self.verbose = verbose
        make_folder(self.basepath)

    def savefig(
        self,
        fig,
        name,
        tight_layout=False,
        basepath=None,
        label_figs=None,
        fileformats=None,
        text_on_top=True,
        **kwargs,
    ):
        """
        Function to save a figure in one or multiple formats.

        Args:
            fig (matplotlib.figure.Figure): The figure to save.
            name (str): The file name to save the figure under.
            tight_layout (bool, optional): If True, the layout is adjusted to fit tightly within the figure. Defaults to False.
            basepath (str, optional): The base path for saving the figure. If None, uses self.basepath. Defaults to None.
            label_figs (list of axes, optional): List of axes to label. If None, no axes are labeled. Defaults to None.
            fileformats (list of str, optional): List of file formats to save the figure in. If None, uses self.fileformats. Defaults to None.
            text_on_top (bool, optional): If True, the text is placed on top of the figure. Defaults to True.
            **kwargs: Additional keyword arguments for the labelfigs function.
        """

        if tight_layout:
            fig.tight_layout()

        if text_on_top:
            bring_text_to_front(fig)

        if basepath is None:
            basepath = self.basepath

        if label_figs is not None:
            for i, ax in enumerate(label_figs):
                labelfigs(ax, i, **kwargs)

        if fileformats is None:
            fileformats = self.fileformats

        for fileformat in fileformats:
            if self.verbose:
                print(basepath + name + "." + fileformat)
            fig.savefig(
                basepath + name + "." + fileformat,
                dpi=self.dpi,
                bbox_inches="tight",
            )
