import pytest
from m3util.viz.lines import handle_linewidth_conflicts, draw_lines
import matplotlib.pyplot as plt


def test_handle_linewidth_conflicts_both_linewidth_and_lw():
    textprops = {"linewidth": 2, "lw": 3}
    updated_props = handle_linewidth_conflicts(textprops)
    assert updated_props["linewidth"] == 3
    assert "lw" not in updated_props


def test_handle_linewidth_conflicts_only_lw():
    textprops = {"lw": 4}
    updated_props = handle_linewidth_conflicts(textprops)
    assert updated_props["linewidth"] == 4
    assert "lw" not in updated_props


def test_handle_linewidth_conflicts_only_linewidth():
    textprops = {"linewidth": 5}
    updated_props = handle_linewidth_conflicts(textprops)
    assert updated_props["linewidth"] == 5
    assert "lw" not in updated_props


def test_handle_linewidth_conflicts_no_linewidth_or_lw():
    textprops = {"color": "blue"}
    updated_props = handle_linewidth_conflicts(textprops)
    assert updated_props == {"color": "blue"}


def test_handle_linewidth_conflicts_linestyle_conflict():
    textprops = {"linestyle": "dotted", "ls": "--"}
    updated_props = handle_linewidth_conflicts(textprops)
    assert updated_props["linestyle"] == "solid"
    assert "ls" not in updated_props


def test_draw_lines_with_style_and_halo():
    fig, ax = plt.subplots()
    x_values = [0, 1, 2]
    y_values = [0, 1, 4]
    style = {"color": "blue", "linewidth": 2, "linestyle": "--"}
    halo = {"enabled": True, "color": "red", "scale": 3}

    line = draw_lines(ax, x_values, y_values, style=style, halo=halo)

    assert len(line) == 1
    assert line[0].get_color() == "blue"
    assert line[0].get_linestyle() == "--"


def test_draw_lines_without_style():
    fig, ax = plt.subplots()
    x_values = [0, 1, 2]
    y_values = [0, 1, 4]

    line = draw_lines(ax, x_values, y_values)

    assert len(line) == 1
    assert line[0].get_color() == "#ff7f0e"  # Default color


def test_draw_lines_empty_y_values():
    fig, ax = plt.subplots()
    x_values = [0, 1, 2]
    y_values = []

    with pytest.raises(
        ValueError,
        match="x and y must have same first dimension, *",
    ):
        draw_lines(ax, x_values, y_values)


def test_draw_lines_invalid_style():
    fig, ax = plt.subplots()
    x_values = [0, 1, 2]
    y_values = [0, 1, 4]
    invalid_style = {"invalid_key": "value"}

    with pytest.raises(
        AttributeError,
        match=r"Line2D\.set\(\) got an unexpected keyword argument 'invalid_key'",
    ):
        draw_lines(ax, x_values, y_values, style=invalid_style)


def test_draw_lines_no_ax():
    x_values = [0, 1, 2]
    y_values = [0, 1, 4]

    with pytest.raises(
        AttributeError, match="'NoneType' object has no attribute 'plot'"
    ):
        draw_lines(None, x_values, y_values)
