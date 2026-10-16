"""Tests for analyze_xodr helpers."""

from __future__ import annotations

import math

import pytest

from autoware_lanelet2_to_opendrive.analyze_xodr import _evaluate_road_xy
from autoware_lanelet2_to_opendrive.opendrive.geometry import Arc, Line, PlanView
from autoware_lanelet2_to_opendrive.opendrive.road import Road


class TestEvaluateRoadXy:
    """``_evaluate_road_xy`` must evaluate line and arc geometry, not only
    paramPoly3 (#502).

    The analyze/QC path reconstructs roads from the XODR; once
    ``parse_roads_from_xodr`` keeps ``<arc>`` and ``<line>`` segments, the
    Frenet-to-world evaluator has to handle them too — previously it read
    paramPoly3 coefficients directly and would fail on other primitives.
    """

    def test_arc_geometry_evaluated_along_its_curve(self) -> None:
        # Quarter circle of radius 10 m, starting at the origin heading +x.
        radius = 10.0
        curvature = 1.0 / radius
        length = radius * math.pi / 2.0
        road = Road(
            id=1,
            plan_view=PlanView(
                geometries=[
                    Arc(
                        s=0.0,
                        x=0.0,
                        y=0.0,
                        hdg=0.0,
                        length=length,
                        curvature=curvature,
                    )
                ]
            ),
        )

        # A point partway along the arc lies on the analytic circle.
        p = length / 3.0
        assert _evaluate_road_xy(road, p) == pytest.approx(
            (
                math.sin(curvature * p) / curvature,
                (1.0 - math.cos(curvature * p)) / curvature,
            ),
            abs=1e-6,
        )
        # The quarter circle ends exactly at (radius, radius).
        assert _evaluate_road_xy(road, length) == pytest.approx(
            (radius, radius), abs=1e-6
        )

    def test_line_geometry_with_lateral_offset(self) -> None:
        # A 20 m line heading +x from the origin.
        road = Road(
            id=2,
            plan_view=PlanView(
                geometries=[Line(s=0.0, x=0.0, y=0.0, hdg=0.0, length=20.0)]
            ),
        )

        assert _evaluate_road_xy(road, 5.0) == pytest.approx((5.0, 0.0), abs=1e-6)
        # Positive t is to the left of travel; heading +x -> left is +y.
        assert _evaluate_road_xy(road, 5.0, 3.0) == pytest.approx((5.0, 3.0), abs=1e-6)


class TestFrameFromGeoReference:
    """``_frame_from_geo_reference`` recovers the offset used at conversion.

    PR #27 changed the MGRS / MGRS+offset geoReference to a ``+proj=tmerc``
    string whose ``+lat_0``/``+lon_0`` are the equator and the UTM central
    meridian, with the frame origin encoded in ``+x_0``/``+y_0``. Reading
    ``lat_0``/``lon_0`` as the origin then yields an offset ~96 km off, and
    every stop line / traffic light fails cross-validation in
    ``carla-import-test``.
    """

    # Offset from conf/map/nishishinjuku.yaml (mgrs_grid: 54SUE).
    NISHISHINJUKU_GRID = "54SUE"
    NISHISHINJUKU_OFFSET = (81655.73, 50137.43)

    @staticmethod
    def _frame(proj_string: str):  # type: ignore[no-untyped-def]
        from autoware_lanelet2_to_opendrive.analyze_xodr import (
            _frame_from_geo_reference,
        )

        frame = _frame_from_geo_reference(proj_string)
        assert frame is not None
        return frame

    @staticmethod
    def _mgrs_forward(lat: float, lon: float) -> tuple[float, float]:
        import lanelet2
        from autoware_lanelet2_extension_python.projection import MGRSProjector

        from autoware_lanelet2_to_opendrive.projection import (
            latlon_to_lanelet2_origin,
        )

        projector = MGRSProjector(latlon_to_lanelet2_origin(lat, lon))
        p = projector.forward(lanelet2.core.GPSPoint(lat, lon, 0.0))
        return p.x, p.y

    def test_mgrs_offset_frame_recovers_conversion_offset(self) -> None:
        from autoware_lanelet2_to_opendrive.projection import (
            mgrs_grid_with_offset_to_proj_string,
        )

        proj = mgrs_grid_with_offset_to_proj_string(
            self.NISHISHINJUKU_GRID, *self.NISHISHINJUKU_OFFSET
        )
        frame = self._frame(proj)
        assert frame.offset_x == pytest.approx(self.NISHISHINJUKU_OFFSET[0], abs=1e-6)
        assert frame.offset_y == pytest.approx(self.NISHISHINJUKU_OFFSET[1], abs=1e-6)
        # The origin handed to MGRSProjector must lie at the offset point,
        # in the same grid square (1 m MGRS precision is enough for that).
        fx, fy = self._mgrs_forward(frame.origin_lat, frame.origin_lon)
        assert fx == pytest.approx(self.NISHISHINJUKU_OFFSET[0], abs=1.5)
        assert fy == pytest.approx(self.NISHISHINJUKU_OFFSET[1], abs=1.5)

    @pytest.mark.parametrize("grid", ["54SUE", "56HLH"])
    def test_mgrs_grid_frame_has_zero_offset(self, grid: str) -> None:
        """Plain mgrs_grid (and lat_lon) origins export with a zero offset."""
        from autoware_lanelet2_to_opendrive.projection import mgrs_to_proj_string

        frame = self._frame(mgrs_to_proj_string(grid))
        assert frame.offset_x == pytest.approx(0.0, abs=1e-6)
        assert frame.offset_y == pytest.approx(0.0, abs=1e-6)

    def test_southern_hemisphere_offset_frame(self) -> None:
        from autoware_lanelet2_to_opendrive.projection import (
            mgrs_grid_with_offset_to_proj_string,
        )

        proj = mgrs_grid_with_offset_to_proj_string("56HLH", 12345.67, 89012.34)
        frame = self._frame(proj)
        assert frame.offset_x == pytest.approx(12345.67, abs=1e-6)
        assert frame.offset_y == pytest.approx(89012.34, abs=1e-6)
        assert frame.origin_lat < 0.0

    @pytest.mark.parametrize(
        "proj",
        [
            # Pre-#27 MGRS+offset string: lat_0/lon_0 is the real origin.
            "+proj=utm +zone=54 +lat_0=35.68 +lon_0=139.69 "
            "+datum=WGS84 +units=m +no_defs",
            # TransverseMercator (#24): lat_0/lon_0 is the real origin.
            "+proj=tmerc +lat_0=35.68 +lon_0=139.69 +k_0=0.9999 "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
            # Pre-#27 TransverseMercator spelling (+k instead of +k_0).
            "+proj=tmerc +lat_0=35.68 +lon_0=139.69 +k=0.9996 "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
        ],
    )
    def test_real_origin_strings_keep_legacy_behaviour(self, proj: str) -> None:
        frame = self._frame(proj)
        assert (frame.origin_lat, frame.origin_lon) == (35.68, 139.69)
        assert (frame.offset_x, frame.offset_y) == self._mgrs_forward(35.68, 139.69)

    def test_unparseable_string_returns_none(self) -> None:
        from autoware_lanelet2_to_opendrive.analyze_xodr import (
            _frame_from_geo_reference,
        )

        assert _frame_from_geo_reference("+proj=longlat +datum=WGS84") is None
