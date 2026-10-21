from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


EXPECTED_EXACT = {
    "numpy": "2.3.5",
    "pandas": "2.2.3",
    "scikit-learn": "1.9.0",
    "joblib": "1.5.3",
    "catboost": "1.2.8",
    "fastapi": "0.141.1",
    "starlette": "1.6.0",
    "pydantic": "2.13.4",
    "uvicorn": "0.52.2",
    "httpx2": "2.10.0",
    "ruff": "0.16.2",
}

IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "ruff": None,  # Ruff is validated as an installed CLI distribution, not imported.
}


def main() -> None:
    print("=== M9 PINNED ENVIRONMENT ===")
    mismatches: list[str] = []
    for package, expected in EXPECTED_EXACT.items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Required M9 package is not installed: {package}") from exc
        marker = "PASS" if actual == expected else "MISMATCH"
        print(f"{package:14s} {actual:12s} expected={expected:12s} {marker}")
        if actual != expected:
            mismatches.append(f"{package}: installed={actual}, expected={expected}")

    if mismatches:
        raise RuntimeError(
            "M9 direct dependency pins are not reproduced. Run `make install-dev`. "
            + "; ".join(mismatches)
        )

    # Import explicitly so broken native/binary wheels fail before tests or image builds.
    for package in EXPECTED_EXACT:
        module = IMPORT_NAMES.get(package, package.replace("-", "_"))
        if module is not None:
            import_module(module)
    print("imports: PASS")
    print("direct pins: PASS")


if __name__ == "__main__":
    main()
