"""Registry for loaded RTA type curves."""

from __future__ import annotations

from src.rta_type_curves.loader import TypeCurveLoader
from src.rta_type_curves.models import RTATypeCurveMethod, TypeCurve


class TypeCurveRegistry:
    """In-memory registry for RTA type curves."""

    def __init__(self, curves: list[TypeCurve] | None = None) -> None:
        self._curves: dict[tuple[RTATypeCurveMethod, str], TypeCurve] = {}
        if curves:
            for curve in curves:
                self.register(curve)

    @classmethod
    def from_loader(
        cls,
        loader: TypeCurveLoader | None = None,
        allow_demo_fallback: bool = True,
    ) -> "TypeCurveRegistry":
        """Create a registry from the default loader."""
        curve_loader = loader or TypeCurveLoader()
        return cls(curve_loader.load_available(allow_demo_fallback=allow_demo_fallback))

    def register(self, curve: TypeCurve) -> None:
        """Register or replace one type curve."""
        key = (curve.method, curve.curve_id)
        self._curves[key] = curve

    def list_methods(self) -> list[RTATypeCurveMethod]:
        """List methods available in the registry."""
        return sorted({method for method, _ in self._curves}, key=lambda method: method.value)

    def list_curve_ids(self, method: RTATypeCurveMethod | str | None = None) -> list[str]:
        """List curve IDs, optionally filtered by method."""
        if method is None:
            return sorted(curve_id for _, curve_id in self._curves)

        method_value = RTATypeCurveMethod(method)
        return sorted(
            curve_id
            for curve_method, curve_id in self._curves
            if curve_method == method_value
        )

    def get(self, method: RTATypeCurveMethod | str, curve_id: str) -> TypeCurve:
        """Get a curve by method and curve_id."""
        method_value = RTATypeCurveMethod(method)
        key = (method_value, curve_id)

        try:
            return self._curves[key]
        except KeyError as exc:
            available = self.list_curve_ids(method_value)
            raise KeyError(
                f"Type curve not found: method={method_value.value}, curve_id={curve_id}. "
                f"Available curve IDs for method: {available}"
            ) from exc

    def get_by_method(self, method: RTATypeCurveMethod | str) -> list[TypeCurve]:
        """Return all curves for a method."""
        method_value = RTATypeCurveMethod(method)
        return [
            curve
            for (curve_method, _), curve in sorted(
                self._curves.items(),
                key=lambda item: (item[0][0].value, item[0][1]),
            )
            if curve_method == method_value
        ]

    def all(self) -> list[TypeCurve]:
        """Return all registered curves."""
        return [
            curve
            for _, curve in sorted(
                self._curves.items(),
                key=lambda item: (item[0][0].value, item[0][1]),
            )
        ]