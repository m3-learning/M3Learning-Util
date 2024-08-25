import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from m3util.util.IO import make_folder
from m3util.viz.layout import labelfigs
from m3util.viz.printing import printer  # Replace 'your_module' with the actual module name

@pytest.fixture
def mock_make_folder():
    with patch('m3util.util.IO.make_folder') as mock:
        yield mock

@pytest.fixture
def mock_labelfigs():
    with patch('m3util.viz.layout.labelfigs') as mock:
        yield mock
        
def test_printer_init(tmp_path, mock_make_folder):
    basepath = str(tmp_path / "test")
    p = printer(dpi=300, basepath=basepath, fileformats=["jpg", "pdf"], verbose=False)
    
    assert p.dpi == 300
    assert p.basepath == basepath
    assert p.fileformats == ["jpg", "pdf"]
    assert p.verbose == False

def test_printer_savefig(tmp_path, mock_make_folder, mock_labelfigs):
    basepath = str(tmp_path / "test") + "/"
    p = printer(dpi=300, basepath=basepath, fileformats=["png"], verbose=True)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    with patch('matplotlib.figure.Figure.savefig') as mock_savefig, \
         patch('builtins.print') as mock_print:
        p.savefig(fig, "test_figure", tight_layout=True, label_figs=[ax])
    
    mock_savefig.assert_called_once_with(
        f"{basepath}test_figure.png",
        dpi=300,
        bbox_inches="tight"
    )
    mock_print.assert_called_once_with(f"{basepath}test_figure.png")

def test_printer_savefig_custom_basepath(tmp_path, mock_make_folder):
    basepath = str(tmp_path / "test")
    custom_basepath = str(tmp_path / "custom")
    p = printer(dpi=300, basepath=basepath, fileformats=["png"], verbose=False)
    
    fig = plt.figure()
    
    with patch('matplotlib.figure.Figure.savefig') as mock_savefig:
        p.savefig(fig, "test_figure", basepath=custom_basepath)
    
    mock_savefig.assert_called_once_with(
        f"{custom_basepath}test_figure.png",
        dpi=300,
        bbox_inches="tight"
    )

def test_printer_savefig_custom_fileformats(tmp_path, mock_make_folder):
    basepath = str(tmp_path / "test") + "/"
    p = printer(dpi=300, basepath=basepath, fileformats=["png"], verbose=False)
    
    fig = plt.figure()
    
    with patch('matplotlib.figure.Figure.savefig') as mock_savefig:
        p.savefig(fig, "test_figure", fileformats=["jpg", "pdf"])
    
    assert mock_savefig.call_count == 2
    mock_savefig.assert_any_call(
        f"{basepath}test_figure.jpg",
        dpi=300,
        bbox_inches="tight"
    )
    mock_savefig.assert_any_call(
        f"{basepath}test_figure.pdf",
        dpi=300,
        bbox_inches="tight"
    )