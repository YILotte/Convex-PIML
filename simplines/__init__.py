from simplines import bsplines
from simplines import cad
from simplines import spaces
from simplines import linalg
from simplines import quadratures
from simplines import utilities
from simplines import api
from simplines import results

__all__ = ['bsplines', 'cad',
           'spaces', 'linalg',
           'quadratures', 'utilities', 'api']

from simplines.bsplines import ( find_span,
                                 basis_funs,
                                 basis_funs_1st_der,
                                 basis_funs_all_ders,
                                 collocation_matrix,
                                 histopolation_matrix,
                                 breakpoints,
                                 greville,
                                 elements_spans,
                                 make_knots,
                                 elevate_knots,
                                 quadrature_grid,
                                 basis_integrals,
                                 basis_ders_on_quad_grid,
                                 scaling_matrix,
                                 hrefinement_matrix )

from simplines.cad import ( point_on_bspline_curve,
                            point_on_nurbs_curve,
                            insert_knot_bspline_curve,
                            insert_knot_nurbs_curve,
                            elevate_degree_bspline_curve,
                            elevate_degree_nurbs_curve,
                            point_on_bspline_surface,
                            point_on_nurbs_surface,
                            insert_knot_bspline_surface,
                            insert_knot_nurbs_surface,
                            elevate_degree_bspline_surface,
                            elevate_degree_nurbs_surface,
                            translate_bspline_curve,
                            translate_nurbs_curve,
                            rotate_bspline_curve,
                            rotate_nurbs_curve,
                            homothetic_bspline_curve,
                            homothetic_nurbs_curve,
                            translate_bspline_surface,
                            translate_nurbs_surface,
                            homothetic_nurbs_curve,
                            translate_bspline_surface,
                            translate_nurbs_surface,
                            rotate_bspline_surface,
                            rotate_nurbs_surface,
                            homothetic_bspline_surface,
                            homothetic_nurbs_surface )

from simplines.spaces import ( SplineSpace,
                               TensorSpace )

from simplines.linalg import ( StencilVectorSpace,
                               StencilVector,
                               StencilMatrix )

from simplines.quadratures import gauss_legendre

from simplines.utilities import ( plot_field_1d,
                                  plot_field_2d,
                                  prolongation_matrix )
from simplines.results import ( sol_field_2d)
from simplines.api import (assemble_matrix, assemble_vector, assemble_scalar, compile_kernel, apply_dirichlet)
