import pytest
from unittest.mock import patch
from m3util.viz.style import set_style

def test_set_style_default():
    with patch("seaborn.reset_orig") as mock_reset_orig, \
         patch("matplotlib.pyplot.rcParams", new_callable=dict) as mock_rcparams, \
         patch("builtins.print") as mock_print:

        # Call the function
        set_style("default")

        # Check that seaborn.reset_orig() was called
        mock_reset_orig.assert_called_once()

        # Check that the correct rcParams were set for default style
        assert mock_rcparams["image.cmap"] == "magma"
        assert mock_rcparams["axes.labelsize"] == 18
        assert mock_rcparams["xtick.labelsize"] == 16
        assert mock_rcparams["ytick.labelsize"] == 16
        assert mock_rcparams["figure.titlesize"] == 20
        assert mock_rcparams["xtick.direction"] == "in"
        assert mock_rcparams["ytick.direction"] == "in"
        assert mock_rcparams["xtick.top"] == True
        assert mock_rcparams["ytick.right"] == True

        # Check that the print statement was called with the expected message
        mock_print.assert_any_call("default set for seaborn")
        mock_print.assert_any_call("default set for matplotlib")

def test_set_style_printing():
    with patch("seaborn.reset_orig") as mock_reset_orig, \
         patch("matplotlib.pyplot.rcParams", new_callable=dict) as mock_rcparams, \
         patch("builtins.print") as mock_print:

        # Call the function
        set_style("printing")

        # Check that seaborn.reset_orig() was called
        mock_reset_orig.assert_called_once()

        # Check that the correct rcParams were set for printing style
        assert mock_rcparams["image.cmap"] == "viridis"
        assert mock_rcparams["axes.labelsize"] == 6
        assert mock_rcparams["xtick.labelsize"] == 5
        assert mock_rcparams["ytick.labelsize"] == 5
        assert mock_rcparams["figure.titlesize"] == 8
        assert mock_rcparams["xtick.direction"] == "in"
        assert mock_rcparams["ytick.direction"] == "in"
        assert mock_rcparams["xtick.top"] == True
        assert mock_rcparams["ytick.right"] == True
        assert mock_rcparams["lines.markersize"] == 0.5
        assert mock_rcparams["axes.grid"] == False
        assert mock_rcparams["lines.linewidth"] == 0.5
        assert mock_rcparams["axes.linewidth"] == 0.5
        assert mock_rcparams["legend.fontsize"] == 5
        assert mock_rcparams["legend.loc"] == "upper left"
        assert mock_rcparams["legend.frameon"] == False

        # Check that the print statement was called with the expected message
        mock_print.assert_any_call("printing set for seaborn")
