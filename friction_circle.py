# -*- coding: utf-8 -*-
"""摩擦圆（联合纵-侧向力）拟合工具 — 基于 tyres.ini 参数。

用法:
    python friction_circle.py [tyres.ini] [-o 输出图片.png] [--fz-mult 0.6 0.8 1.0 1.2 1.4]

根据 tyres.ini 中的注释参数建立 AC 刷模型 + 摩擦椭圆联合滑移模型，并绘制:
  1. 摩擦椭圆随载荷的变化（理想抓地极限包络）
  2. 纯滑移力曲线（Fx-kappa 与 Fy-alpha，标出峰值 mu*Fz）
  3. 联合滑移力矢场（同时施加纵向+侧向滑移时的力向量）
  4. 模型可达包络 vs 理想摩擦椭圆（拟合对比）
  5. 纵向力 Fx 被侧向滑移削弱（不同 alpha 下 Fx-kappa 曲线）
  6. 侧向力 Fy 被纵向滑移削弱（不同 kappa 下 Fy-alpha 曲线）

模型说明（与 tyres.ini 注释对应）:
  - 峰值摩擦系数随载荷（幂律）:
        mu_y(Fz) = DY_REF * (Fz/FZ0)^(LS_EXPY-1)
        mu_x(Fz) = DX_REF * (Fz/FZ0)^(LS_EXPX-1)
    （若文件指定了 DY_CURVE/DX_CURVE 且旁边有对应 LUT，本脚本暂不解析，
     仍使用 DY_REF/DX_REF 幂律近似，见控制台提示。）
  - 纯滑移形状: AC 刷模型（本项目 AC_to_SVJ 基准公式）
        S(u) = (2u/(1+u^2))*bell + FALLOFF_LEVEL*(1-bell)*tanh(1.5u)
        bell = exp(-(u-1)^2/sigma^2), sigma = clip(1.6/sqrt(FALLOFF_SPEED), 0.25, 4)
  - 峰值滑移参考: 侧向峰值出现在 FRICTION_LIMIT_ANGLE(度) 处；
    纵向刚度 = CX_MULT * 侧向刚度（见 tyres.ini 的 CX_MULT 注释），
    因此纵向峰值滑移 kappa_peak = FRICTION_LIMIT_ANGLE / CX_MULT（弧度当量）。
  - 联合滑移（摩擦圆）: 归一化滑移 u_x=|kappa|/kappa_peak, u_y=|alpha|/alpha_peak，
    两个方向的力被对方滑移按 COMBINED_FACTOR 削弱:
        Fx = Fx0 / (1 + CF*uy^2),   Fy = Fy0 / (1 + CF*ux^2)
    再施加摩擦椭圆硬约束:
        (Fx/(mu_x*Fz))^2 + (Fy/(mu_y*Fz))^2 <= 1   （超限按比例缩回椭圆上）
  - XMU 是 CSP 扩展物理的联合滑移系数，AC 经典模型不直接使用（图中以文字注明）。
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（Windows 优先 Microsoft YaHei / SimHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# tyres.ini 解析
# ---------------------------------------------------------------------------
def parse_tyres_ini(path: Path) -> dict[str, dict[str, str]]:
    """解析 AC tyres.ini，返回 {大节名: {KEY: value}}。自动忽略 ; 注释。"""
    sections: dict[str, dict[str, str]] = {}
    current = ""
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            current = m.group(1).strip().upper()
            sections.setdefault(current, {})
            continue
        if current and "=" in line:
            key, value = (p.strip() for p in line.split("=", 1))
            sections[current][key.upper()] = value
    return sections


def get_float(section: dict[str, str], key: str, default: float) -> float:
    try:
        return float(section[key])
    except (KeyError, ValueError):
        return default


def build_params(path: Path) -> dict:
    """从 tyres.ini 提取摩擦圆拟合所需参数（取 FRONT 节，前后轴同配方）。"""
    sections = parse_tyres_ini(path)
    sec = sections.get("FRONT", {})
    p = {
        "name": sec.get("NAME", "Unknown"),
        "short": sec.get("SHORT_NAME", "TYRE")[:4],
        "fz0": get_float(sec, "FZ0", 1000.0),
        "dy_ref": get_float(sec, "DY_REF", get_float(sec, "DY0", 1.0)),
        "dx_ref": get_float(sec, "DX_REF", get_float(sec, "DX0", 1.0)),
        "ls_expy": get_float(sec, "LS_EXPY", 1.0),
        "ls_expx": get_float(sec, "LS_EXPX", 1.0),
        "friction_limit_angle_deg": get_float(sec, "FRICTION_LIMIT_ANGLE", 8.0),
        "cx_mult": get_float(sec, "CX_MULT", 1.0),
        "falloff_level": get_float(sec, "FALLOFF_LEVEL", 0.75),
        "falloff_speed": get_float(sec, "FALLOFF_SPEED", 2.0),
        "combined_factor": get_float(sec, "COMBINED_FACTOR", 1.0),
        "xmu": get_float(sec, "XMU", 0.25),
        "flex": get_float(sec, "FLEX", 0.0005),
        "flex_gain": get_float(sec, "FLEX_GAIN", 0.0),
    }
    for key in ("DY_CURVE", "DX_CURVE"):
        p[key] = sec.get(key)
    return p


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class TractionCircleModel:
    """AC 刷模型 + 摩擦椭圆联合滑移模型。"""

    def __init__(self, p: dict) -> None:
        self.p = p
        self.alpha_peak = math.radians(p["friction_limit_angle_deg"])  # 侧向峰值滑移角 (rad)
        # CX_MULT = 纵向刚度/侧向刚度，因此纵向峰值滑移 = 侧向峰值/CX_MULT
        self.kappa_peak = self.alpha_peak / max(p["cx_mult"], 1e-6)
        # 峰值滑移随载荷展宽（FLEX_GAIN: 2 倍 FZ0 时额外叠加的比例）
        self._flex_gain = p["flex_gain"]

    def mu_y(self, fz: float) -> float:
        p = self.p
        return p["dy_ref"] * max(fz / p["fz0"], 1e-6) ** (p["ls_expy"] - 1.0)

    def mu_x(self, fz: float) -> float:
        p = self.p
        return p["dx_ref"] * max(fz / p["fz0"], 1e-6) ** (p["ls_expx"] - 1.0)

    def _peak_slip(self, base_peak: float, fz: float, fz0: float) -> float:
        """峰值滑移随载荷展宽: base * (1 + FLEX_GAIN*(Fz/FZ0 - 1))，下限保持 base。"""
        return base_peak * max(1.0 + self._flex_gain * (fz / fz0 - 1.0), 1.0)

    def brush(self, u: np.ndarray) -> np.ndarray:
        """AC 刷模型饱和形状（含峰值后回落）。"""
        p = self.p
        u = np.maximum(np.asarray(u, dtype=float), 0.0)
        sigma = float(np.clip(1.6 / math.sqrt(max(p["falloff_speed"], 1e-3)), 0.25, 4.0))
        bell = np.exp(-((u - 1.0) ** 2) / (sigma ** 2))
        pure = 2.0 * u / (1.0 + u * u)
        return pure * bell + p["falloff_level"] * (1.0 - bell) * np.tanh(u * 1.5)

    def pure_longitudinal(self, kappa: np.ndarray, fz: float) -> np.ndarray:
        """纯纵向力 Fx(kappa)。"""
        kappa = np.asarray(kappa, dtype=float)
        peak_slip = self._peak_slip(self.kappa_peak, fz, self.p["fz0"])
        u = np.abs(kappa) / max(peak_slip, 1e-9)
        return np.sign(kappa) * self.mu_x(fz) * fz * self.brush(u)

    def pure_lateral(self, alpha: np.ndarray, fz: float) -> np.ndarray:
        """纯侧向力 Fy(alpha)，alpha 单位为弧度。"""
        alpha = np.asarray(alpha, dtype=float)
        peak_slip = self._peak_slip(self.alpha_peak, fz, self.p["fz0"])
        u = np.abs(alpha) / max(peak_slip, 1e-9)
        return np.sign(alpha) * self.mu_y(fz) * fz * self.brush(u)

    def combined(self, kappa: np.ndarray, alpha: np.ndarray, fz: float) -> tuple[np.ndarray, np.ndarray]:
        """联合滑移力 (Fx, Fy)：纵-侧向同时作用时的摩擦圆约束。"""
        p = self.p
        kappa = np.asarray(kappa, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        peak_slip_y = self._peak_slip(self.alpha_peak, fz, p["fz0"])
        peak_slip_x = self._peak_slip(self.kappa_peak, fz, p["fz0"])
        ux = np.abs(kappa) / max(peak_slip_x, 1e-9)
        uy = np.abs(alpha) / max(peak_slip_y, 1e-9)

        fx0 = self.pure_longitudinal(kappa, fz)
        fy0 = self.pure_lateral(alpha, fz)

        # COMBINED_FACTOR 联合滑移耦合: 每个方向的力被对方滑移削弱
        cf = max(p["combined_factor"], 0.0)
        fx = fx0 / (1.0 + cf * uy * uy)
        fy = fy0 / (1.0 + cf * ux * ux)

        # 摩擦椭圆硬约束: (Fx/mx)^2 + (Fy/my)^2 <= 1
        mx = self.mu_x(fz) * fz
        my = self.mu_y(fz) * fz
        r = np.sqrt((fx / mx) ** 2 + (fy / my) ** 2)
        over = r > 1.0
        if np.any(over):
            fx[over] = fx[over] / r[over]
            fy[over] = fy[over] / r[over]
        return fx, fy

    def ellipse(self, fz: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """理想摩擦椭圆边界 (Fx, Fy)。"""
        theta = np.linspace(0.0, 2.0 * np.pi, n)
        fx = self.mu_x(fz) * fz * np.cos(theta)
        fy = self.mu_y(fz) * fz * np.sin(theta)
        return fx, fy


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def make_figure(p: dict, fz_mults: list[float], out_path: Path) -> None:
    m = TractionCircleModel(p)
    fz0 = p["fz0"]
    fzs = [fz0 * mult for mult in fz_mults]

    fig = plt.figure(figsize=(16, 9.5))
    fig.suptitle(
        f"{p['name']}（{p['short']}）摩擦圆拟合 — Traction Circle Fit\n"
        f"FZ0={fz0:.0f} N · μy(DY_REF)={p['dy_ref']:.3f} · μx(DX_REF)={p['dx_ref']:.3f} · "
        f"LS_EXPY={p['ls_expy']:.3f} · LS_EXPX={p['ls_expx']:.3f} · "
        f"CF={p['combined_factor']:.2f} · FRICTION_LIMIT_ANGLE={p['friction_limit_angle_deg']:.0f}° · "
        f"CX_MULT={p['cx_mult']:.2f} · FALLOFF_L={p['falloff_level']:.2f}/S={p['falloff_speed']:.0f}",
        fontsize=11,
    )

    # ---- 1. 摩擦椭圆随载荷 ----
    ax = fig.add_subplot(2, 3, 1)
    cmap = plt.get_cmap("viridis")
    for i, fz in enumerate(fzs):
        fx, fy = m.ellipse(fz)
        ax.plot(fx, fy, color=cmap(i / max(len(fzs) - 1, 1)), lw=1.8,
                label=f"{fz:.0f} N (μy={m.mu_y(fz):.3f}, μx={m.mu_x(fz):.3f})")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.axvline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_title("1. 理想摩擦椭圆随载荷\n(抓地极限包络)")
    ax.set_xlabel("纵向力 Fx (N)")
    ax.set_ylabel("侧向力 Fy (N)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # ---- 2. 纯滑移力曲线 ----
    ax = fig.add_subplot(2, 3, 2)
    fz = fz0
    kappa = np.linspace(-0.30, 0.30, 241)
    alpha = np.linspace(-math.radians(15), math.radians(15), 241)
    fx = m.pure_longitudinal(kappa, fz)
    fy = m.pure_lateral(alpha, fz)
    ax.plot(kappa * 100, fx, color="#1f77b4", lw=2.2, label="Fx (纵向)")
    ax.axhline(m.mu_x(fz) * fz, color="#1f77b4", ls=":", lw=1.2)
    ax.axhline(-m.mu_x(fz) * fz, color="#1f77b4", ls=":", lw=1.2)
    ax.set_xlabel("纵向滑移率 κ (%)")
    ax.set_ylabel("Fx (N)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(np.degrees(alpha), fy, color="#d62728", lw=2.2, label="Fy (侧向)")
    ax2.axhline(m.mu_y(fz) * fz, color="#d62728", ls=":", lw=1.2)
    ax2.axhline(-m.mu_y(fz) * fz, color="#d62728", ls=":", lw=1.2)
    ax2.set_xlabel("侧偏角 α (deg)")
    ax2.set_ylabel("Fy (N)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title(f"2. 纯滑移力曲线 @ {fz:.0f} N\n(虚线 = 峰值 μ·Fz)")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    # ---- 3. 联合滑移力矢场 ----
    ax = fig.add_subplot(2, 3, 3)
    fz = fz0
    kappa_g = np.linspace(-0.30, 0.30, 49)
    alpha_g = np.linspace(-math.radians(14), math.radians(14), 49)
    kg, ag = np.meshgrid(kappa_g, alpha_g)
    fxg, fyg = m.combined(kg.ravel(), ag.ravel(), fz)
    total = np.hypot(fxg, fyg)
    sc = ax.scatter(fxg, fyg, c=total, cmap="plasma", s=9, alpha=0.85)
    fx_e, fy_e = m.ellipse(fz)
    ax.plot(fx_e, fy_e, color="k", lw=2.0, ls="--", label="理想摩擦椭圆")
    ax.plot(*m.ellipse(fz * 0.6), color="0.5", lw=1.0, ls="--")
    ax.plot(*m.ellipse(fz * 1.4), color="0.5", lw=1.0, ls="--")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("合力 |F| (N)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.axvline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_title(f"3. 联合滑移力矢场 @ {fz:.0f} N\n(κ×α 网格同时施加)")
    ax.set_xlabel("Fx (N)")
    ax.set_ylabel("Fy (N)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # ---- 4. 模型可达包络 vs 理想椭圆 ----
    ax = fig.add_subplot(2, 3, 4)
    fz = fz0
    kappa_d = np.linspace(-0.35, 0.35, 121)
    alpha_d = np.linspace(-math.radians(16), math.radians(16), 121)
    kd, ad = np.meshgrid(kappa_d, alpha_d)
    fxd, fyd = m.combined(kd.ravel(), ad.ravel(), fz)
    ax.scatter(fxd, fyd, s=2.5, alpha=0.30, color="#2ca02c", label="模型可达力点（拟合包络）")
    fx_e, fy_e = m.ellipse(fz)
    ax.plot(fx_e, fy_e, color="k", lw=2.2, label="理想摩擦椭圆")
    # 摩擦圆（等 μ 参考）
    ax.plot(m.mu_x(fz) * fz * np.cos(np.linspace(0, 2 * np.pi, 200)),
            m.mu_x(fz) * fz * np.sin(np.linspace(0, 2 * np.pi, 200)),
            color="0.6", lw=1.0, ls=":", label="等 μx 摩擦圆参考")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.axvline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_title(f"4. 模型可达包络 vs 理想椭圆 @ {fz:.0f} N\n(拟合对比)")
    ax.set_xlabel("Fx (N)")
    ax.set_ylabel("Fy (N)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # ---- 5. 纵向力被侧向滑移削弱 ----
    ax = fig.add_subplot(2, 3, 5)
    fz = fz0
    kappa_c = np.linspace(-0.30, 0.30, 241)
    for alpha_deg in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
        fx_c, _ = m.combined(kappa_c, np.full_like(kappa_c, math.radians(alpha_deg)), fz)
        ax.plot(kappa_c * 100, fx_c, lw=1.8, label=f"α = {alpha_deg:.0f}°")
    ax.set_title("5. 纵向力 Fx 被侧向滑移削弱\n(Fx-κ，不同固定 α)")
    ax.set_xlabel("纵向滑移率 κ (%)")
    ax.set_ylabel("Fx (N)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---- 6. 侧向力被纵向滑移削弱 ----
    ax = fig.add_subplot(2, 3, 6)
    fz = fz0
    alpha_c = np.linspace(-math.radians(14), math.radians(14), 241)
    for kappa_v in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        _, fy_c = m.combined(np.full_like(alpha_c, kappa_v), alpha_c, fz)
        ax.plot(np.degrees(alpha_c), fy_c, lw=1.8, label=f"κ = {kappa_v:.2f}")
    ax.set_title("6. 侧向力 Fy 被纵向滑移削弱\n(Fy-α，不同固定 κ)")
    ax.set_xlabel("侧偏角 α (deg)")
    ax.set_ylabel("Fy (N)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.text(
        0.995, 0.005,
        f"模型: AC 刷模型 + 摩擦椭圆约束 (Fx/(μx·Fz))²+(Fy/(μy·Fz))²≤1；"
        f"COMBINED_FACTOR 联合滑移耦合 CF={p['combined_factor']:.2f}；"
        f"XMU={p['xmu']:.2f}（CSP 联合滑移系数，AC 模型未直接使用）；"
        f"μ(Fz) 用 DY_REF/DX_REF 幂律近似" + ("；DY_CURVE/DX_CURVE 存在但未解析" if p.get("DY_CURVE") else ""),
        ha="right", va="bottom", fontsize=8, color="0.35",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] 已生成: {out_path}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="基于 tyres.ini 的摩擦圆（联合纵-侧向力）拟合")
    ap.add_argument("ini", nargs="?", default="tyres.ini", help="tyres.ini 路径（默认 ./tyres.ini）")
    ap.add_argument("-o", "--output", default=None, help="输出 PNG 路径")
    ap.add_argument("--fz-mult", nargs="+", type=float, default=[0.6, 0.8, 1.0, 1.2, 1.4],
                    help="椭圆图的载荷倍数（相对 FZ0），默认 0.6 0.8 1.0 1.2 1.4")
    args = ap.parse_args()

    ini = Path(args.ini)
    if not ini.is_file():
        raise SystemExit(f"找不到 tyres.ini: {ini}")
    p = build_params(ini)
    if p["DY_CURVE"] or p["DX_CURVE"]:
        print(f"[提示] 文件引用了 DY_CURVE={p.get('DY_CURVE')} / DX_CURVE={p.get('DX_CURVE')}，"
              f"本脚本未解析 LUT，μ(Fz) 使用 DY_REF/DX_REF 幂律近似。")
    print(f"参数: {p['name']} ({p['short']})  FZ0={p['fz0']:.0f} N  "
          f"μy(FZ0)={p['dy_ref']:.3f}  μx(FZ0)={p['dx_ref']:.3f}  "
          f"α_peak={p['friction_limit_angle_deg']:.1f}°  "
          f"κ_peak={p['friction_limit_angle_deg'] / max(p['cx_mult'], 1e-9):.3f}°-当量")
    out = Path(args.output) if args.output else Path(f"friction_circle_{p['short'].lower()}.png")
    make_figure(p, args.fz_mult, out)


if __name__ == "__main__":
    main()
