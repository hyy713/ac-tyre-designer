from pathlib import Path

import numpy as np

from ac_tyre_designer.model import (
    TyreDefinition, cornering_stiffness_csp, cornering_stiffness_mf,
    csp_force, export_ac_package, fit_csp, load_unitire_definition, magic_formula,
    render_tyres_ini, validate_tyres_ini,
)


def test_magic_formula_is_odd_at_default_shifts():
    tyre = TyreDefinition()
    x = np.array([-0.1, 0.0, 0.1])
    y = magic_formula(x, tyre.reference_load_n, tyre.reference_load_n, tyre.lateral)
    assert np.isclose(y[0], -y[2])
    assert y[1] == 0


def test_fit_and_render_contains_csp_keys():
    tyre = TyreDefinition()
    fit = fit_csp(tyre)
    text = render_tyres_ini(tyre, fit)
    for key in ("DY0=", "DX0=", "LS_EXPY=", "LS_EXPX=", "DY_REF=", "DX_REF=", "FALLOFF_LEVEL=", "[THERMAL_FRONT]"):
        assert key in text
    assert "[FRONT]\n" in text
    assert "[FRONT_0]" not in text
    validate_tyres_ini(text)
    assert fit.lateral.r_squared > 0.8
    assert fit.longitudinal.r_squared > 0.8
    assert 0.65 <= fit.lateral.falloff_level <= 0.98
    assert 1.0 <= fit.lateral.falloff_speed <= 7.0
    assert fit.longitudinal.stiffness / 18.0 <= 2.0


def test_ac_transient_and_structural_tuning_values_are_exported():
    tyre = TyreDefinition(
        relaxation_length_m=0.5393,
        flex=0.00056,
        flex_gain=0.0265,
        friction_limit_angle_deg=16.0,
    )
    text = render_tyres_ini(tyre, fit_csp(tyre))
    assert "RELAXATION_LENGTH=0.5393000" in text
    assert "FLEX=0.0005600" in text
    assert "FLEX_GAIN=0.0265000" in text
    assert "FRICTION_LIMIT_ANGLE=16.000000" in text


def test_tyres_ini_parameters_are_annotated():
    """Every exported parameter block carries a Chinese explanatory comment."""
    text = render_tyres_ini(TyreDefinition(), fit_csp(TyreDefinition()))
    for marker in (
        "; NAME:", "; DY0:", "; DX0:", "; FZ0:", "; LS_EXPY:", "; FALLOFF_LEVEL:",
        "; CX_MULT:", "; PRESSURE_STATIC:", "; CAMBER_GAIN:", "; COMBINED_FACTOR:",
        "; RELAXATION_LENGTH:", "; SURFACE_TRANSFER:", "; FRICTION_K:",
        "wear_curve.lut", "thermal_performance.lut",
    ):
        assert marker in text
    assert "MZ" not in text


def test_structural_tyre_values_are_not_derived_from_reference_load():
    tyre = TyreDefinition(
        angular_inertia_kgm2=0.12,
        tyre_damping_ns_m=750.0,
        tyre_rate_n_m=50388.0,
        reference_load_n=1200.0,
    )
    text = render_tyres_ini(tyre, fit_csp(tyre))
    assert "ANGULAR_INERTIA=0.120000" in text
    assert "DAMP=750.000" in text
    assert "RATE=50388.000" in text
    assert "FZ0=1200.000" in text


def test_import_supplied_measured_unitire_model():
    model_path = Path(__file__).parent / "data" / "hoosier_43105_unitire_model.json"
    tyre = load_unitire_definition(model_path)
    assert 800.0 < tyre.reference_load_n < 820.0
    assert 100000.0 < tyre.tyre_rate_n_m < 130000.0
    assert np.isclose(tyre.relaxation_length_m, 0.7084406020680589)
    assert tyre.lateral.mu0 > 2.0
    assert tyre.longitudinal.mu0 > 2.0


def test_ac_forward_model_is_zero_at_zero_slip():
    tyre = TyreDefinition()
    fit = fit_csp(tyre)
    force = csp_force(np.array([0.0]), tyre.reference_load_n, tyre.reference_load_n, fit.lateral, "lateral")
    assert force[0] == 0


def test_cornering_stiffness_matches_magic_formula_origin_slope():
    tyre = TyreDefinition()
    stiffness = cornering_stiffness_mf(tyre.reference_load_n, tyre)
    expected = (tyre.reference_load_n * tyre.lateral.mu0
                * tyre.lateral.B * tyre.lateral.C)
    assert np.isclose(stiffness, expected, rtol=1e-6)
    fit = fit_csp(tyre)
    assert cornering_stiffness_csp(
        tyre.reference_load_n, tyre.reference_load_n, fit.lateral
    ) > 0


def test_export_package(tmp_path: Path):
    tyre = TyreDefinition()
    generated = export_ac_package(tmp_path, tyre, fit_csp(tyre))
    assert len(generated) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in generated)
    wear = (tmp_path / "wear_curve.lut").read_text(encoding="utf-8")
    assert "0.3|100" in wear
    assert max(float(line.split("|")[1]) for line in wear.splitlines()) == 100
    assert {"tyres.ini", "wear_curve.lut", "thermal_performance.lut", "tyre_design.json"}.issubset(
        {p.name for p in tmp_path.iterdir()})
