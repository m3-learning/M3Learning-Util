import sys

if sys.version_info[:2] >= (3, 11):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    dist_name = "M3Learning-Util"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
    
from m3util import converters
from m3util import globus
from m3util import ml
from m3util import notebooks
from m3util import pandas
from m3util import util
from m3util import viz

from m3util.converters import (complex, to_complex,)
from m3util.globus import (GlobusAccessError, check_globus_endpoint,
                           check_globus_file_access, globus,)
from m3util.ml import (AdaHessian, ContrastiveLoss, DivergenceLoss,
                       GlobalScaler, SimpleModel4, Sparse_Max_Loss, TRCG,
                       TrustRegion, Weighted_LN_loss, closure_fn3, computeTime,
                       inference, logging, optimizers, preprocessor, rand,
                       rand_tensor, regularization, save_list_to_txt,
                       set_seeds, test_trcg_step_shrink_radius, write_csv,)
from m3util.notebooks import (add_tags_to_notebook,
                              calculate_notebook_checksum, checksum,
                              convert_notebook_to_slides, main, skip_execution,
                              slides,)
from m3util.pandas import (filter, filter_df, find_min_max_by_group,)
from m3util.util import (IO, append_to_csv, calculate_h5file_checksum, code,
                         compress_folder, download, download_and_unzip,
                         download_file, download_files_from_txt,
                         extract_number, filter_cls_params, filter_kwargs,
                         find_files_recursive, find_groups_with_string,
                         find_measurement, generate_init, generate_init_py,
                         get_size, get_tree, get_tuple_names, h5, hashing,
                         in_list, kwargs, main, make_dataset, make_folder,
                         make_group, print_code, print_tree, reporthook,
                         search, select_hash_algorithm, unzip,)
from m3util.viz import (Axis_Ratio, DrawArrow, FigDimConverter, Path,
                        PathPatch, add_box, add_colorbar, add_scalebar,
                        add_text_to_figure, arrows, bring_text_to_front,
                        colorbars, combine_lines, display_image,
                        draw_ellipse_with_arrow, draw_extended_arrow_indicator,
                        draw_line_with_text, draw_lines, embedding_maps,
                        find_nearest, get_axis_pos_inches, get_axis_range,
                        get_closest_point, get_perpendicular_vector,
                        get_zorders, handle_linewidth_conflicts, imagemap,
                        images, inset_connector, labelfigs, layout, layout_fig,
                        layout_subfigures_inches, line_annotation, lines,
                        make_movie, mock_line_annotation, movies,
                        number_to_letters, obj_offset, path_maker,
                        place_text_in_inches, place_text_points,
                        plot_into_graph, positioning, printer, printing,
                        scalebar, set_axis, set_sci_notation_label, set_style,
                        shift_object_in_inches, shift_object_in_points,
                        span_to_axis, style, subfigures, text,)

__all__ = ['AdaHessian', 'Axis_Ratio', 'ContrastiveLoss', 'DivergenceLoss',
           'DrawArrow', 'FigDimConverter', 'GlobalScaler', 'GlobusAccessError',
           'IO', 'Path', 'PathPatch', 'SimpleModel4', 'Sparse_Max_Loss',
           'TRCG', 'TrustRegion', 'Weighted_LN_loss', 'add_box',
           'add_colorbar', 'add_scalebar', 'add_tags_to_notebook',
           'add_text_to_figure', 'append_to_csv', 'arrows',
           'bring_text_to_front', 'calculate_h5file_checksum',
           'calculate_notebook_checksum', 'check_globus_endpoint',
           'check_globus_file_access', 'checksum', 'closure_fn3', 'code',
           'colorbars', 'combine_lines', 'complex', 'compress_folder',
           'computeTime', 'convert_notebook_to_slides', 'converters',
           'display_image', 'download', 'download_and_unzip', 'download_file',
           'download_files_from_txt', 'draw_ellipse_with_arrow',
           'draw_extended_arrow_indicator', 'draw_line_with_text',
           'draw_lines', 'embedding_maps', 'extract_number', 'filter',
           'filter_cls_params', 'filter_df', 'filter_kwargs',
           'find_files_recursive', 'find_groups_with_string',
           'find_measurement', 'find_min_max_by_group', 'find_nearest',
           'generate_init', 'generate_init_py', 'get_axis_pos_inches',
           'get_axis_range', 'get_closest_point', 'get_perpendicular_vector',
           'get_size', 'get_tree', 'get_tuple_names', 'get_zorders', 'globus',
           'h5', 'handle_linewidth_conflicts', 'hashing', 'imagemap', 'images',
           'in_list', 'inference', 'inset_connector', 'kwargs', 'labelfigs',
           'layout', 'layout_fig', 'layout_subfigures_inches',
           'line_annotation', 'lines', 'logging', 'main', 'make_dataset',
           'make_folder', 'make_group', 'make_movie', 'ml',
           'mock_line_annotation', 'movies', 'notebooks', 'number_to_letters',
           'obj_offset', 'optimizers', 'pandas', 'path_maker',
           'place_text_in_inches', 'place_text_points', 'plot_into_graph',
           'positioning', 'preprocessor', 'print_code', 'print_tree',
           'printer', 'printing', 'rand', 'rand_tensor', 'regularization',
           'reporthook', 'save_list_to_txt', 'scalebar', 'search',
           'select_hash_algorithm', 'set_axis', 'set_sci_notation_label',
           'set_seeds', 'set_style', 'shift_object_in_inches',
           'shift_object_in_points', 'skip_execution', 'slides',
           'span_to_axis', 'style', 'subfigures',
           'test_trcg_step_shrink_radius', 'text', 'to_complex', 'unzip',
           'util', 'viz', 'write_csv']
