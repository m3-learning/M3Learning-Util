import pytest
import matplotlib.pyplot as plt
from m3util.viz.positioning import obj_offset

def test_obj_offset_no_offset():
    # Test when no offset is given (default behavior)
    xy = (1, 1)
    new_xy = obj_offset(xy)
    assert new_xy == xy


def test_obj_offset_with_fontsize_units_right():
    # Test offset with fontsize units to the right
    xy = (1, 1)
    new_xy = obj_offset(xy, offset="right", offset_units="fontsize")
    fontsize = plt.rcParams["font.size"] / 2 * 1.2
    assert new_xy == (xy[0] + fontsize, xy[1])


def test_obj_offset_with_fontsize_units_up():
    # Test offset with fontsize units upwards
    xy = (1, 1)
    new_xy = obj_offset(xy, offset="up", offset_units="fontsize")
    fontsize = plt.rcParams["font.size"] / 2 * 1.2
    assert new_xy == (xy[0], xy[1] + fontsize)


def test_obj_offset_with_inches_units():
    # Test offset with inches units
    fig, ax = plt.subplots()
    xy = (1, 1)
    new_xy = obj_offset(xy, offset=(0.1, 0.2), offset_units="inches", ax=ax)
    dpi = fig.dpi
    assert new_xy == (xy[0] + 0.1 * dpi, xy[1] + 0.2 * dpi)


def test_obj_offset_with_points_units():
    # Test offset with points units
    xy = (1, 1)
    new_xy = obj_offset(xy, offset=(10, 20), offset_units="points")
    assert new_xy == (xy[0] + 10, xy[1] + 20)


def test_obj_offset_with_fraction_units():
    # Test offset with fraction units
    fig, ax = plt.subplots()
    xy = (1, 1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    new_xy = obj_offset(xy, offset=(0.1, 0.2), offset_units="fraction", ax=ax)
    assert new_xy == (
        xy[0] + 1,
        xy[1] + 2,
    )  # 0.1 of x range (10-0) and 0.2 of y range (10-0)


def test_obj_offset_invalid_units():
    # Test with invalid units
    xy = (1, 1)
    with pytest.raises(
        ValueError, match="Units must be 'fontsize', 'inches', 'points', or 'fraction'."
    ):
        obj_offset(xy, offset=(1, 1), offset_units="invalid_unit")


def test_obj_offset_missing_ax_for_inches():
    # Test missing ax argument when offset_units is 'inches'
    xy = (1, 1)
    with pytest.raises(
        ValueError,
        match="Please provide an axes object when using 'inches' or 'fraction' units.",
    ):
        obj_offset(xy, offset=(0.1, 0.2), offset_units="inches")


def test_obj_offset_missing_ax_for_fraction():
    # Test missing ax argument when offset_units is 'fraction'
    xy = (1, 1)
    with pytest.raises(
        ValueError,
        match="Please provide an axes object when using 'inches' or 'fraction' units.",
    ):
        obj_offset(xy, offset=(0.1, 0.2), offset_units="fraction")

