from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.io import loadmat


Axis = Literal["lateral", "longitudinal"]


@dataclass
class MagicFormulaAxis:
    """Compact pure-slip Magic Formula definition.

    The peak coefficient varies with load as mu=mu0*(Fz/Fz0)**load_exp.
    B, C and E define shape; horizontal/vertical shifts use the same units as
    the input slip and normalized force respectively.
    """

    B: float
    C: float
    E: float
    mu0: float
    load_exp: float
    shift_x: float = 0.0
    shift_y: float = 0.0


@dataclass
class TyreDefinition:
    name: str = "Custom CSP Tyre"
    short_name: str = "CSP"
    width_m: float = 0.1905
    radius_m: float = 0.2286
    rim_radius_m: float = 0.1397
    angular_inertia_kgm2: float = 0.12
    tyre_damping_ns_m: float = 750.0
    tyre_rate_n_m: float = 50388.0
    reference_load_n: float = 1200.0
    pressure_static_psi: float = 12.0
    pressure_ideal_psi: float = 17.0
    relaxation_length_m: float = 0.50
    flex: float = 0.00056
    flex_gain: float = 0.0265
    friction_limit_angle_deg: float = 11.0
    lateral: MagicFormulaAxis = None  # type: ignore[assignment]
    longitudinal: MagicFormulaAxis = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.lateral is None:
            self.lateral = MagicFormulaAxis(10.0, 1.30, -0.5, 1.35, -0.08)
        if self.longitudinal is None:
            self.longitudinal = MagicFormulaAxis(12.0, 1.45, -0.2, 1.40, -0.07)


@dataclass
class CSPAxisFit:
    mu0: float
    mu1: float
    load_exp: float
    stiffness: float
    falloff_level: float
    falloff_speed: float
    rmse: float = 0.0
    r_squared: float = 0.0


@dataclass
class CSPFit:
    lateral: CSPAxisFit
    longitudinal: CSPAxisFit


def magic_formula(slip: np.ndarray, load_n: float, ref_load_n: float,
                  p: MagicFormulaAxis) -> np.ndarray:
    x = np.asarray(slip, dtype=float) + p.shift_x
    mu = p.mu0 * max(load_n / ref_load_n, 1e-6) ** p.load_exp
    bx = p.B * x
    shape = np.sin(p.C * np.arctan(bx - p.E * (bx - np.arctan(bx))))
    return load_n * (mu * shape + p.shift_y)


AC_FLEX = 0.00055


def slip_grid(axis: Axis) -> np.ndarray:
    return np.linspace(-0.35, 0.35, 281) if axis == "longitudinal" else np.deg2rad(np.linspace(-18, 18, 289))


def _csp_shape(normalized_slip: np.ndarray, falloff_level: float,
               falloff_speed: float) -> np.ndarray:
    """AC/CSP brush saturation curve used by the public AC_to_SVJ bench."""
    u = np.maximum(np.asarray(normalized_slip, dtype=float), 0.0)
    sigma = float(np.clip(1.6 / math.sqrt(max(falloff_speed, 1e-3)), 0.25, 4.0))
    bell = np.exp(-((u - 1.0) ** 2) / (sigma ** 2))
    pure = 2.0 * u / (1.0 + u * u)
    return pure * bell + falloff_level * (1.0 - bell) * np.tanh(u * 1.5)


def csp_force(slip: np.ndarray, load_n: float, ref_load_n: float,
              p: CSPAxisFit, axis: Axis) -> np.ndarray:
    """Evaluate the AC-style pure-slip forward model."""
    slip = np.asarray(slip, dtype=float)
    ratio = max(load_n / ref_load_n, 1e-6)
    mu = max(0.05, p.mu0 + p.mu1 * (ratio - 1.0)) * ratio ** (p.load_exp - 1.0)
    peak = mu * load_n
    slip_stiffness = p.stiffness * load_n / (1.0 + AC_FLEX * load_n)
    peak_slip = peak / max(slip_stiffness, 1e-6)
    normalized = np.abs(slip) / max(peak_slip, 1e-5)
    return np.sign(slip) * peak * _csp_shape(normalized, p.falloff_level, p.falloff_speed)


def cornering_stiffness_mf(load_n: float, tyre: TyreDefinition) -> float:
    """Return target lateral stiffness dFy/dalpha at zero, in N/rad."""
    step = 1e-6
    force = magic_formula(
        np.array([-step, step]), load_n, tyre.reference_load_n, tyre.lateral
    )
    return float((force[1] - force[0]) / (2.0 * step))


def cornering_stiffness_csp(load_n: float, ref_load_n: float,
                            fit: CSPAxisFit) -> float:
    """Return fitted AC/CSP lateral stiffness at zero, in N/rad."""
    step = 1e-6
    force = csp_force(
        np.array([-step, step]), load_n, ref_load_n, fit, "lateral"
    )
    return float((force[1] - force[0]) / (2.0 * step))


def slip_grid(axis: Axis) -> np.ndarray:
    return np.linspace(-0.35, 0.35, 281) if axis == "longitudinal" else np.deg2rad(np.linspace(-18, 18, 289))


def fit_axis(mf: MagicFormulaAxis, ref_load_n: float, axis: Axis) -> CSPAxisFit:
    grid = slip_grid(axis)
    loads = ref_load_n * np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    target = np.concatenate([magic_formula(grid, fz, ref_load_n, mf) / fz for fz in loads])

    def residual(v: np.ndarray) -> np.ndarray:
        fit = CSPAxisFit(*v, 0.0, 0.0)
        pred = np.concatenate([csp_force(grid, fz, ref_load_n, fit, axis) / fz for fz in loads])
        # Peak region receives slightly more influence without ignoring tails.
        return (pred - target) * (1.0 + 0.35 * np.abs(target))

    peak_slip = abs(grid[np.argmax(np.abs(magic_formula(grid, ref_load_n, ref_load_n, mf)))])
    stiffness0 = max(mf.mu0 * (1.0 + AC_FLEX * ref_load_n) / max(peak_slip, 1e-3), 1.0)
    stiffness_max = 200.0 if axis == "lateral" else 36.0  # CX_MULT <= 2.0
    stiffness0 = min(stiffness0, stiffness_max * 0.95)
    initial = np.array([mf.mu0, 0.0, 1.0 + mf.load_exp, stiffness0, 0.85, 3.0])
    lower = np.array([0.2, -0.5, 0.35, 0.5, 0.65, 1.0])
    upper = np.array([3.0, 0.25, 1.30, stiffness_max, 0.98, 7.0])
    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=3000)
    fitted = CSPAxisFit(*map(float, result.x), 0.0, 0.0)
    pred = np.concatenate([csp_force(grid, fz, ref_load_n, fitted, axis) / fz for fz in loads])
    error = pred - target
    fitted.rmse = float(np.sqrt(np.mean(error ** 2)))
    denom = float(np.sum((target - target.mean()) ** 2))
    fitted.r_squared = 1.0 - float(np.sum(error ** 2)) / denom if denom else 1.0
    return fitted


def fit_csp(tyre: TyreDefinition) -> CSPFit:
    lateral = fit_axis(tyre.lateral, tyre.reference_load_n, "lateral")
    longitudinal = fit_axis(tyre.longitudinal, tyre.reference_load_n, "longitudinal")
    shared_level = (lateral.falloff_level + longitudinal.falloff_level) / 2.0
    shared_speed = (lateral.falloff_speed + longitudinal.falloff_speed) / 2.0
    lateral.falloff_level = longitudinal.falloff_level = shared_level
    lateral.falloff_speed = longitudinal.falloff_speed = shared_speed
    return CSPFit(lateral=lateral, longitudinal=longitudinal)


#: 设计 JSON 的字段说明（JSON 不支持注释，导出时写入顶层 _notes 键）。
#: 注意：csp_fit 包含 lateral/longitudinal 两个子对象，字段含义相同。
_FIELD_NOTES: dict[str, dict[str, str]] = {
    "说明": "本文件由 AC CSP Tyre Designer 生成，可在工具中“打开设计”重新编辑。单位与 tyres.ini 一致。",
    "tyre": {
        "name": "轮胎名称",
        "short_name": "短名称（游戏内显示）",
        "width_m": "轮胎宽度（米）",
        "radius_m": "未加载自由半径（米）",
        "rim_radius_m": "轮辋半径（米）",
        "angular_inertia_kgm2": "轮胎转动惯量（kg·m²）",
        "tyre_damping_ns_m": "轮胎垂向阻尼（N·s/m）",
        "tyre_rate_n_m": "轮胎垂向刚度（N/m）",
        "reference_load_n": "参考载荷 FZ0（N）",
        "pressure_static_psi": "静态胎压（psi）",
        "pressure_ideal_psi": "理想胎压（psi）",
        "relaxation_length_m": "松弛长度（米），瞬态响应距离常数",
        "flex": "胎体侧向柔性系数",
        "flex_gain": "胎体柔性随载荷的增益",
        "friction_limit_angle_deg": "摩擦极限角（度）",
        "lateral": "侧向 Magic Formula：B/C/E 形状系数，mu0 峰值摩擦系数，load_exp 载荷指数，shift_x/shift_y 水平/垂直偏移",
        "longitudinal": "纵向 Magic Formula（含义同侧向，方向为纵向）",
    },
    "csp_fit": {
        "lateral": "侧向 AC/CSP 反向拟合结果（字段含义见下）",
        "longitudinal": "纵向 AC/CSP 反向拟合结果（字段含义见下）",
        "mu0": "峰值摩擦系数基准值（对应 tyres.ini 的 DY0/DX0）",
        "mu1": "摩擦系数随载荷的线性斜率（DY1/DX1）",
        "load_exp": "载荷指数（LS_EXPY/LS_EXPX）",
        "stiffness": "峰值滑移刚度（CX_MULT = stiffness/18）",
        "falloff_level": "峰值后残余力水平（FALLOFF_LEVEL）",
        "falloff_speed": "峰值后回落速度（FALLOFF_SPEED）",
        "rmse": "拟合归一化均方根误差（越小越好）",
        "r_squared": "拟合决定系数 R²（越接近 1 越好）",
    },
}


def save_definition(path: Path, tyre: TyreDefinition, fit: CSPFit | None = None) -> None:
    payload = {
        "tyre": asdict(tyre),
        "csp_fit": asdict(fit) if fit else None,
        "_notes": _FIELD_NOTES,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_definition(path: Path) -> TyreDefinition:
    payload = json.loads(path.read_text(encoding="utf-8"))["tyre"]
    payload["lateral"] = MagicFormulaAxis(**payload["lateral"])
    payload["longitudinal"] = MagicFormulaAxis(**payload["longitudinal"])
    # Drop the legacy aligning-moment block from designs saved before it was removed.
    payload.pop("aligning", None)
    return TyreDefinition(**payload)


def _unitire_pure_force(slip: np.ndarray, load_n: float, ref_load_n: float,
                        force: dict[str, float], axis: Axis) -> np.ndarray:
    """Evaluate a pure-slip slice of the measured UniTire JSON model."""
    ratio_delta = load_n / ref_load_n - 1.0
    if axis == "lateral":
        mu0, mu_slope = force["muy0"], force["muyLoadSlope"]
        stiffness0, stiffness_slope = force["Cy0"], force["CyLoadSlope"]
        slip_term = np.tan(slip)
    else:
        mu0, mu_slope = force["mux0"], force["muxLoadSlope"]
        stiffness0, stiffness_slope = force["Cx0"], force["CxLoadSlope"]
        slip_term = slip
    mu = mu0 * max(0.25, 1.0 + mu_slope * ratio_delta)
    stiffness = stiffness0 * load_n * max(0.15, 1.0 + stiffness_slope * ratio_delta)
    capacity = max(mu * load_n, 1.0)
    u = stiffness * np.asarray(slip_term, dtype=float)
    rho = np.maximum(np.abs(u) / capacity, 1e-9)
    curvature = force["curvatureE"]
    saturation = 1.0 - np.exp(
        -rho - curvature * rho ** 2 - (curvature ** 2 + 1.0 / 12.0) * rho ** 3
    )
    saturation = np.clip(saturation, -1.35, 1.35)
    return capacity * saturation * np.sign(u)


def _fit_magic_formula_to_unitire(force: dict[str, float], ref_load_n: float,
                                   axis: Axis) -> MagicFormulaAxis:
    grid = slip_grid(axis)
    loads = np.linspace(200.0, 1200.0, 6)
    target = np.concatenate([
        _unitire_pure_force(grid, fz, ref_load_n, force, axis) / fz for fz in loads
    ])

    def residual(values: np.ndarray) -> np.ndarray:
        mf = MagicFormulaAxis(*map(float, values))
        prediction = np.concatenate([
            magic_formula(grid, fz, ref_load_n, mf) / fz for fz in loads
        ])
        return prediction - target

    mu_key = "muy0" if axis == "lateral" else "mux0"
    stiffness_key = "Cy0" if axis == "lateral" else "Cx0"
    shape_c = 1.30 if axis == "lateral" else 1.45
    initial_b = force[stiffness_key] / max(force[mu_key] * shape_c, 1e-6)
    initial = np.array([initial_b, shape_c, 0.0, force[mu_key], -0.10, 0.0, 0.0])
    lower = np.array([0.1, 0.5, -5.0, 0.2, -1.0, -0.05, -0.05])
    upper = np.array([50.0, 2.5, 2.0, 4.0, 1.0, 0.05, 0.05])
    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=4000)
    return MagicFormulaAxis(*map(float, result.x))


def _estimate_ttc_vertical_rate(model_path: Path, payload: dict) -> float | None:
    """Estimate dFz/d(deflection) from stationary spring-rate blocks near P0."""
    root = model_path.parent.parent
    tire_folder = root / "tire"
    p0 = float(payload["force"]["P0_kPa"])
    run_files = payload.get("trainingSummary", {}).get("runs", [])
    estimates: list[float] = []
    for filename in run_files:
        mat_path = tire_folder / filename
        if not mat_path.is_file():
            continue
        data = loadmat(mat_path, squeeze_me=True)
        required = ("V", "SA", "IA", "P", "FZ", "RL")
        if any(key not in data for key in required):
            continue
        speed, slip_angle = np.ravel(data["V"]), np.ravel(data["SA"])
        camber, pressure = np.ravel(data["IA"]), np.ravel(data["P"])
        load, loaded_radius = -np.ravel(data["FZ"]), np.ravel(data["RL"]) / 100.0
        mask = (np.abs(speed) < 0.8) & (np.abs(slip_angle) < 0.25) & \
               (np.abs(camber) < 0.25) & (np.abs(pressure - p0) < 2.0) & \
               (load > 100.0) & (load < 1400.0) & np.isfinite(loaded_radius)
        if np.count_nonzero(mask) < 100:
            continue
        slope = np.polyfit(loaded_radius[mask], load[mask], 1)[0]
        if slope < 0:
            estimates.append(float(-slope))
    return float(np.median(estimates)) if estimates else None


def load_unitire_definition(path: Path) -> TyreDefinition:
    """Convert the measured UniTire JSON format into an editable AC tyre design."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    force = payload["force"]
    ref_load = float(force["Fz0_N"])
    name = str(payload.get("name", "Measured UniTire"))
    pressure_psi = float(force["P0_kPa"]) / 6.894757293
    loaded_radius = float(payload.get("fitOptions", {}).get(
        "LoadedRadiusM", payload.get("my", {}).get("loadedRadius_m", 0.215)))
    # 18.0 x 7.5-10 sizing from the supplied TTC model. Fall back to its
    # measured loaded radius if a future JSON does not use this naming style.
    radius = 0.2286 if "18.0" in name else loaded_radius * 1.06
    width = 0.1905 if "7.5" in name else 0.19
    rim_radius = 0.1397 if " 10 " in f" {name} " else max(radius * 0.60, 0.05)
    vertical_rate = _estimate_ttc_vertical_rate(path, payload) or 116000.0
    transient = payload.get("transient", {})
    return TyreDefinition(
        name=name, short_name="R25B", width_m=width, radius_m=radius,
        rim_radius_m=rim_radius, angular_inertia_kgm2=0.12,
        tyre_damping_ns_m=750.0, tyre_rate_n_m=vertical_rate,
        reference_load_n=ref_load, pressure_static_psi=pressure_psi,
        pressure_ideal_psi=pressure_psi,
        relaxation_length_m=float(transient.get("sigmaY0_m", 0.5)),
        flex=0.00056, flex_gain=0.0265, friction_limit_angle_deg=16.0,
        lateral=_fit_magic_formula_to_unitire(force, ref_load, "lateral"),
        longitudinal=_fit_magic_formula_to_unitire(force, ref_load, "longitudinal"),
    )


def _axis_comment(label: str, p: CSPAxisFit) -> str:
    return f"; {label} fit: R2={p.r_squared:.6f}, normalized RMSE={p.rmse:.6f}"


def render_tyres_ini(tyre: TyreDefinition, fit: CSPFit) -> str:
    """Render a complete CSP-ready single-compound tyres.ini.

    Front and rear sections intentionally share the designed compound. Users
    can independently tune/copy a second definition after initial validation.
    """
    lat, lon = fit.lateral, fit.longitudinal
    # fit_axis already converts the editor's mu exponent to AC's force exponent.
    ls_expy = lat.load_exp
    ls_expx = lon.load_exp
    common = f"""; ===== 轮胎基础信息 =====
; NAME: 轮胎名称（仅用于识别，可随意修改）
NAME={tyre.name}
; SHORT_NAME: 短名称（游戏内显示，最多 4 个字符）
SHORT_NAME={tyre.short_name[:4].upper()}
; TYPE_HINT: 轮胎类型提示（SLICK = 光头胎/热熔胎）
TYPE_HINT=SLICK
; WIDTH: 轮胎宽度（米）
WIDTH={tyre.width_m:.6f}
; RADIUS: 轮胎未加载自由半径（米）
RADIUS={tyre.radius_m:.6f}
; RIM_RADIUS: 轮辋半径（米）
RIM_RADIUS={tyre.rim_radius_m:.6f}
; ANGULAR_INERTIA: 轮胎转动惯量（kg·m²），决定轮速变化的快慢
ANGULAR_INERTIA={tyre.angular_inertia_kgm2:.6f}
; DAMP: 轮胎垂向阻尼（N·s/m）
DAMP={tyre.tyre_damping_ns_m:.3f}
; RATE: 轮胎垂向刚度（N/m），由实测载荷-形变数据拟合得到
RATE={tyre.tyre_rate_n_m:.3f}

; ===== 摩擦系数与载荷特性（由反向拟合得到） =====
; DY0: 侧向峰值摩擦系数基准值（名义载荷 FZ0 下）
DY0={lat.mu0:.7f}
; DY1: 侧向摩擦系数随载荷的线性斜率（mu = DY0 + DY1*(Fz/FZ0-1)）
DY1={lat.mu1:.7f}
; DX0: 纵向峰值摩擦系数基准值
DX0={lon.mu0:.7f}
; DX1: 纵向摩擦系数随载荷的线性斜率
DX1={lon.mu1:.7f}
; FZ0: 参考载荷（N），所有载荷归一化都以它为基准
FZ0={tyre.reference_load_n:.3f}
; LS_EXPY: 侧向力随载荷的幂指数（CSP 力公式中的载荷指数）
LS_EXPY={ls_expy:.7f}
; LS_EXPX: 纵向力随载荷的幂指数
LS_EXPX={ls_expx:.7f}
; DY_REF: 侧向摩擦参考值（CSP 摩擦基准，一般与 DY0 相同）
DY_REF={lat.mu0:.7f}
; DX_REF: 纵向摩擦参考值（一般与 DX0 相同）
DX_REF={lon.mu0:.7f}

; ===== CSP 刷模型形状参数（决定力曲线形状） =====
; FRICTION_LIMIT_ANGLE: 摩擦极限角（度），超过该侧偏角认为进入完全滑移
FRICTION_LIMIT_ANGLE={tyre.friction_limit_angle_deg:.6f}
; XMU: 联合滑移摩擦系数（侧向与纵向联合作用时的摩擦耦合）
XMU=0.25
; FALLOFF_LEVEL: 峰值后的残余力水平（0.65~0.98，越大峰值后回落越少）
FALLOFF_LEVEL={(lat.falloff_level + lon.falloff_level) / 2:.7f}
; FALLOFF_SPEED: 峰值后力回落的快慢（1~7，越大回落越快）
FALLOFF_SPEED={(lat.falloff_speed + lon.falloff_speed) / 2:.7f}
; CX_MULT: 纵向刚度乘子（须 <= 2.0，对应 AC 的纵向刚度系数）
CX_MULT={lon.stiffness / 18.0:.7f}

; ===== 胎压 =====
; PRESSURE_STATIC: 静态胎压（psi）
PRESSURE_STATIC={tyre.pressure_static_psi:.3f}
; PRESSURE_IDEAL: 理想胎压（psi），在此胎压下性能最佳
PRESSURE_IDEAL={tyre.pressure_ideal_psi:.3f}
; PRESSURE_SPRING_GAIN: 胎压变化对垂向刚度的影响增益
PRESSURE_SPRING_GAIN=8000
; PRESSURE_FLEX_GAIN: 胎压变化对胎体柔性 FLEX 的影响增益
PRESSURE_FLEX_GAIN=0.45
; PRESSURE_RR_GAIN: 胎压变化对滚动阻力的影响增益
PRESSURE_RR_GAIN=0.55
; PRESSURE_D_GAIN: 胎压变化对垂向阻尼的影响增益
PRESSURE_D_GAIN=0.004

; ===== 外倾角 =====
; CAMBER_GAIN: 外倾角对侧向力的增益
CAMBER_GAIN=0.10
; DCAMBER_0/DCAMBER_1: 外倾角磨损多项式系数（外倾角对磨损速度的影响）
DCAMBER_0=1.0
DCAMBER_1=-12

; ===== 滚动阻力 =====
; ROLLING_RESISTANCE_0: 滚动阻力基准值
ROLLING_RESISTANCE_0=10
; ROLLING_RESISTANCE_1: 滚动阻力随载荷的系数
ROLLING_RESISTANCE_1=0.0005
; ROLLING_RESISTANCE_SLIP: 滑移对滚动阻力的影响系数
ROLLING_RESISTANCE_SLIP=5000

; ===== 胎体结构与瞬态 =====
; FLEX: 胎体侧向柔性系数（影响侧偏刚度随载荷的饱和）
FLEX={tyre.flex:.7f}
; FLEX_GAIN: 胎体柔性随载荷的增益
FLEX_GAIN={tyre.flex_gain:.7f}
; RADIUS_ANGULAR_K: 转速对有效滚动半径的影响系数
RADIUS_ANGULAR_K=0.02
; BRAKE_DX_MOD: 制动工况下纵向力的修正系数
BRAKE_DX_MOD=0.05
; COMBINED_FACTOR: 联合滑移强度（越大，纵向滑移对侧向力的衰减越强）
COMBINED_FACTOR=2.0
; WEAR_CURVE: 磨损曲线文件名（格式：x=磨损量，y=剩余性能百分比，新胎≈100）
WEAR_CURVE=wear_curve.lut
; SPEED_SENSITIVITY: 摩擦系数随速度的敏感性（越大，高速时抓地衰减越快）
SPEED_SENSITIVITY=0.003
; RELAXATION_LENGTH: 松弛长度（米），瞬态力建立的“记忆距离”，越大响应越慢
RELAXATION_LENGTH={tyre.relaxation_length_m:.7f}
"""
    thermal = f"""; ===== 轮胎热力学模型（胎温对性能的影响） =====
; SURFACE_TRANSFER: 胎面表面与空气的传热系数
SURFACE_TRANSFER=0.018
; PATCH_TRANSFER: 胎面与路面的传热系数
PATCH_TRANSFER=0.00027
; CORE_TRANSFER: 胎体核心与胎面的传热系数
CORE_TRANSFER=0.0005
; INTERNAL_CORE_TRANSFER: 胎体内部核心的传热系数
INTERNAL_CORE_TRANSFER=0.0035
; FRICTION_K: 摩擦产热系数（打滑时产生热量）
FRICTION_K=0.025
; ROLLING_K: 滚动产热系数
ROLLING_K=0.20
; PERFORMANCE_CURVE: 温度性能曲线文件名（格式：x=胎温°C，y=抓地力乘子，1.0=最佳）
PERFORMANCE_CURVE=thermal_performance.lut
; GRAIN_GAMMA/GRAIN_GAIN: 起粒（grain）现象模型参数
GRAIN_GAMMA=1
GRAIN_GAIN=0.0
; BLISTER_GAMMA/BLISTER_GAIN: 起泡（blister）现象模型参数
BLISTER_GAMMA=1
BLISTER_GAIN=0.0
; COOL_FACTOR: 冷却系数（越大降温越快）
COOL_FACTOR=2.0
; SURFACE_ROLLING_K: 表面滚动摩擦产热系数
SURFACE_ROLLING_K=0.9
"""
    return f"""; Generated by AC CSP Tyre Designer
; 需要 CSP Extended Physics 才能正确读取本文件（原版 AC 会忽略部分参数）。
; 反向拟合为近似结果，请在游戏中用 CSP 轮胎调试工具最终验证。
; 本文件每个参数均带注释说明其作用。
; 配套文件说明:
;   wear_curve.lut          : 磨损曲线。x=磨损量（0=全新），y=剩余性能百分比（新胎≈100）。
;   thermal_performance.lut : 温度性能曲线。x=胎温(°C)，y=抓地力乘子（1.0=最佳工作温度）。
;   tyre_design.json        : 设计文件，可重新导入本工具编辑（含全部参数与拟合结果）。
{_axis_comment('lateral', lat)}
{_axis_comment('longitudinal', lon)}

[HEADER]
VERSION=10

[COMPOUND_DEFAULT]
INDEX=0

[VIRTUALKM]
USE_LOAD=1

[FRONT]
{common}

[REAR]
; 后轴目前使用与前轴相同的复合配方（本工具按单复合设计）。
; 如需前后不同的轮胎，导出后把后胎段替换为另一套参数即可。
{common}

[THERMAL_FRONT]
{thermal}

[THERMAL_REAR]
; 后轴热力学参数与前轴相同。
{thermal}
"""


def validate_tyres_ini(text: str) -> None:
    """Reject structural errors that make AC discard the tyre compound."""
    sections: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip().upper()
            if current in sections:
                raise ValueError(f"Duplicate section [{current}] at line {line_number}")
            sections[current] = {}
            continue
        if current is None or "=" not in line:
            raise ValueError(f"Invalid INI syntax at line {line_number}: {raw}")
        key, value = (part.strip() for part in line.split("=", 1))
        key = key.upper()
        if not key or not value:
            raise ValueError(f"Empty key or value at line {line_number}: {raw}")
        if key in sections[current]:
            raise ValueError(f"Duplicate key [{current}] {key} at line {line_number}")
        sections[current][key] = value

    required_sections = {"HEADER", "COMPOUND_DEFAULT", "FRONT", "REAR", "THERMAL_FRONT", "THERMAL_REAR"}
    missing_sections = sorted(required_sections - sections.keys())
    if missing_sections:
        raise ValueError(f"Missing required sections: {', '.join(missing_sections)}")
    tyre_keys = {
        "NAME", "SHORT_NAME", "WIDTH", "RADIUS", "RIM_RADIUS", "ANGULAR_INERTIA",
        "DAMP", "RATE", "DY0", "DY1", "DX0", "DX1", "WEAR_CURVE", "FZ0",
        "LS_EXPY", "LS_EXPX", "DY_REF", "DX_REF", "FRICTION_LIMIT_ANGLE",
        "PRESSURE_STATIC", "PRESSURE_IDEAL",
    }
    for section in ("FRONT", "REAR"):
        missing_keys = sorted(tyre_keys - sections[section].keys())
        if missing_keys:
            raise ValueError(f"[{section}] missing keys: {', '.join(missing_keys)}")
        values = sections[section]
        numeric_ranges = {
            "FALLOFF_LEVEL": (0.60, 0.99), "FALLOFF_SPEED": (1.0, 7.0),
            "CX_MULT": (0.20, 2.0), "LS_EXPY": (0.30, 1.30),
            "LS_EXPX": (0.30, 1.30), "FRICTION_LIMIT_ANGLE": (1.0, 25.0),
            "RELAXATION_LENGTH": (0.01, 5.0), "FLEX": (0.00001, 0.01),
            "FLEX_GAIN": (0.0, 1.0),
            "ANGULAR_INERTIA": (0.01, 10.0), "DAMP": (50.0, 3000.0),
            "RATE": (5000.0, 1000000.0), "FZ0": (100.0, 30000.0),
        }
        for key, (minimum, maximum) in numeric_ranges.items():
            value = float(values[key])
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"[{section}] {key}={value:g} is outside the safe AC range "
                    f"{minimum:g}..{maximum:g}"
                )


def export_ac_package(folder: Path, tyre: TyreDefinition, fit: CSPFit,
                      tyres_filename: str = "tyres.ini") -> list[Path]:
    """Export an AC data package and return every generated path."""
    folder.mkdir(parents=True, exist_ok=True)
    tyres_path = folder / tyres_filename
    wear_path = folder / "wear_curve.lut"
    thermal_path = folder / "thermal_performance.lut"
    design_path = folder / "tyre_design.json"
    tyres_text = render_tyres_ini(tyre, fit)
    validate_tyres_ini(tyres_text)
    tyres_path.write_text(tyres_text, encoding="utf-8", newline="\n")
    # AC wear curves return remaining tyre condition in percent, not a 0..1
    # multiplier. Writing 1.0 here makes CSP treat a new tyre as 1% remaining
    # and reduces Fx/Fy to approximately 0.01 of their intended values.
    wear_text = "0|99.5\n0.3|100\n0.6|100\n1.0|99.5\n2|96\n4|95\n8|94\n11|92\n12|88\n"
    wear_values = [float(line.split("|", 1)[1]) for line in wear_text.splitlines()]
    if max(wear_values) < 50.0:
        raise ValueError("Wear LUT must use AC percentage values near 100, not 0..1 multipliers")
    wear_path.write_text(wear_text, encoding="utf-8", newline="\n")
    thermal_path.write_text(
        "0|0.70\n40|0.82\n65|0.96\n80|1.00\n95|1.00\n110|0.96\n130|0.85\n160|0.65\n", encoding="utf-8")
    save_definition(design_path, tyre, fit)
    generated = [tyres_path, wear_path, thermal_path, design_path]
    missing = [path for path in generated if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise OSError(f"Export verification failed: {', '.join(str(p) for p in missing)}")
    return generated
