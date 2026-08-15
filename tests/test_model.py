from pathlib import Path

import numpy as np

from ac_tyre_designer.model import (
    TyreDefinition, combined_force, cornering_stiffness_csp, cornering_stiffness_mf,
    csp_force, csp_peak_force, export_ac_package, fit_csp, friction_ellipse,
    load_unitire_definition, magic_formula, render_tyres_ini, validate_tyres_ini,
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


def test_combined_force_friction_circle():
    tyre = TyreDefinition()
    fit = fit_csp(tyre)
    fz = tyre.reference_load_n

    # zero slip -> zero force
    fx, fy = combined_force(np.array([0.0]), np.array([0.0]), fz, fz, fit, tyre)
    assert fx[0] == 0 and fy[0] == 0

    # pure lateral reproduced when kappa = 0
    alpha = np.deg2rad(np.linspace(-10.0, 10.0, 21))
    _, fy_c = combined_force(np.zeros_like(alpha), alpha, fz, fz, fit, tyre)
    assert np.allclose(fy_c, csp_force(alpha, fz, fz, fit.lateral, "lateral"), atol=1e-9)

    # pure longitudinal reproduced when alpha = 0
    kappa = np.linspace(-0.3, 0.3, 21)
    fx_c, _ = combined_force(kappa, np.zeros_like(kappa), fz, fz, fit, tyre)
    assert np.allclose(fx_c, csp_force(kappa, fz, fz, fit.longitudinal, "longitudinal"), atol=1e-9)

    # friction ellipse constraint is never exceeded
    kg, ag = np.meshgrid(np.linspace(-0.5, 0.5, 31), np.deg2rad(np.linspace(-20.0, 20.0, 31)))
    fxg, fyg = combined_force(kg.ravel(), ag.ravel(), fz, fz, fit, tyre)
    mx = csp_peak_force(fit.longitudinal, fz, fz)
    my = csp_peak_force(fit.lateral, fz, fz)
    r = np.sqrt((fxg / mx) ** 2 + (fyg / my) ** 2)
    assert r.max() <= 1.0 + 1e-9

    # combined slip degrades both components vs their pure-slip values
    fx_c2, fy_c2 = combined_force(np.array([0.2]), np.deg2rad(np.array([8.0])), fz, fz, fit, tyre)
    fx0 = csp_force(np.array([0.2]), fz, fz, fit.longitudinal, "longitudinal")
    fy0 = csp_force(np.deg2rad(np.array([8.0])), fz, fz, fit.lateral, "lateral")
    assert abs(fx_c2[0]) < abs(fx0[0])
    assert abs(fy_c2[0]) < abs(fy0[0])

    # ellipse geometry: pure-axis intercepts equal mu*Fz (sampling tolerance)
    fx_e, fy_e = friction_ellipse(fz, fz, fit)
    assert np.isclose(fx_e.max(), mx, rtol=1e-4)
    assert np.isclose(fy_e.max(), my, rtol=1e-4)


def test_combined_factor_is_exported_and_editable():
    assert TyreDefinition().combined_factor == 2.0
    tyre = TyreDefinition(combined_factor=2.24)
    text = render_tyres_ini(tyre, fit_csp(tyre))
    assert "COMBINED_FACTOR=2.2400000" in text
    assert "XMU=0.25" in text


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
