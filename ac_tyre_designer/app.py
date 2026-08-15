from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
from pathlib import Path
import shutil
import traceback

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .model import (
    CSPFit, MagicFormulaAxis, TyreDefinition, combined_force, cornering_stiffness_csp,
    cornering_stiffness_mf, csp_force, export_ac_package, fit_csp, friction_ellipse,
    load_definition, load_unitire_definition, magic_formula, save_definition, slip_grid,
)


# Defaults match the measured model: Hoosier 18.0 x 7.5 10 R25B (Item 43105),
# full-data UniTire fit, see
# codex_tiremodle/unitire_output_full/hoosier_43105_unitire_model.json.
FIELDS = [
    ("name", "Tyre name", "Hoosier 18.0 x 7.5 10 R25B (Item 43105)"),
    ("short_name", "Short name", "R25B"),
    ("width_m", "Width (m)", "0.1905"), ("radius_m", "Radius (m)", "0.2286"),
    ("rim_radius_m", "Rim radius (m)", "0.1397"),
    ("angular_inertia_kgm2", "Angular inertia (kg m2)", "0.12"),
    ("tyre_damping_ns_m", "Tyre damping (N s/m)", "750"),
    ("tyre_rate_n_m", "Tyre rate (N/m)", "116106.923"),
    ("reference_load_n", "Reference load (N)", "815.405"),
    ("pressure_static_psi", "Static pressure (psi)", "12.021"),
    ("pressure_ideal_psi", "Ideal pressure (psi)", "12.021"),
]
AC_TUNING_FIELDS = [
    ("relaxation_length_m", "Relaxation length (m)", "0.0959457"),
    ("flex", "Tyre flex", "0.00056"),
    ("flex_gain", "Flex gain", "0.0265"),
    ("friction_limit_angle_deg", "Friction limit angle (deg)", "16.0"),
    ("combined_factor", "Combined slip factor", "2.0"),
]
AXIS_FIELDS = [("B", "B", "10"), ("C", "C", "1.3"), ("E", "E", "-0.5"),
               ("mu0", "Mu at FZ0", "1.35"), ("load_exp", "Load exponent", "-0.08"),
               ("shift_x", "Horizontal shift", "0"), ("shift_y", "Vertical shift", "0")]

SPIN_CONFIG = {
    "width_m": (0.05, 1.0, 0.005), "radius_m": (0.05, 1.5, 0.005),
    "rim_radius_m": (0.05, 1.0, 0.005), "reference_load_n": (100.0, 30000.0, 100.0),
    "angular_inertia_kgm2": (0.01, 10.0, 0.01),
    "tyre_damping_ns_m": (50.0, 3000.0, 10.0), "tyre_rate_n_m": (5000.0, 1000000.0, 1000.0),
    "pressure_static_psi": (1.0, 80.0, 0.5), "pressure_ideal_psi": (1.0, 80.0, 0.5),
    "relaxation_length_m": (0.01, 5.0, 0.01), "flex": (0.00001, 0.01, 0.00001),
    "flex_gain": (0.0, 1.0, 0.001), "friction_limit_angle_deg": (1.0, 30.0, 0.1),
    "combined_factor": (0.0, 10.0, 0.01),
    "B": (0.01, 100.0, 0.1), "C": (0.1, 5.0, 0.01), "E": (-5.0, 5.0, 0.01),
    "mu0": (0.05, 5.0, 0.01), "load_exp": (-2.0, 2.0, 0.01),
    "shift_x": (-1.0, 1.0, 0.001), "shift_y": (-1.0, 1.0, 0.001),
}


class DesignerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AC CSP Tyre Designer 0.3.0")
        self.geometry("1320x900")
        self.minsize(1050, 680)
        self.vars: dict[str, tk.StringVar] = {}
        self.fit: CSPFit | None = None
        self._redraw_job: str | None = None
        self.status = tk.StringVar(value="Enter Magic Formula parameters, then click Fit CSP.")
        self._build_ui()
        self._set_defaults()
        self.after(100, self.redraw)

    def _build_ui(self) -> None:
        root = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(root, padding=4)
        chart = ttk.Frame(root)
        root.add(left, weight=0)
        root.add(chart, weight=1)

        # Keep all actions visible even when the parameter form is taller than
        # the display (for example with 125%/150% Windows scaling).
        actions1 = ttk.Frame(left)
        actions1.pack(fill=tk.X, pady=(0, 3))
        ttk.Button(actions1, text="Update plots", command=self.redraw).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions1, text="Fit CSP", command=self.run_fit).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions1, text="Export to AC", command=self.export_ac).pack(side=tk.LEFT, padx=2)
        actions2 = ttk.Frame(left)
        actions2.pack(fill=tk.X, pady=3)
        ttk.Button(actions2, text="Open design", command=self.open_design).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions2, text="Save design", command=self.save_design).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions2, text="Import measured", command=self.import_measured).pack(side=tk.LEFT, padx=2)
        ttk.Label(left, textvariable=self.status, wraplength=310, foreground="#174a7e").pack(fill=tk.X, padx=2, pady=(3, 6))

        scroll_host = ttk.Frame(left)
        scroll_host.pack(fill=tk.BOTH, expand=True)
        controls_canvas = tk.Canvas(scroll_host, width=320, highlightthickness=0)
        controls_scroll = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=controls_canvas.yview)
        controls_canvas.configure(yscrollcommand=controls_scroll.set)
        controls_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls = ttk.Frame(controls_canvas, padding=8)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind(
            "<Configure>",
            lambda _event: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(controls_window, width=event.width),
        )

        row = 0
        ttk.Label(controls, text="Basic parameters", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6))
        row += 1
        for key, label, _ in FIELDS:
            self.vars[key] = tk.StringVar()
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=2)
            if key in SPIN_CONFIG:
                self._make_spinbox(controls, self.vars[key], key).grid(row=row, column=1, sticky="ew", pady=2)
            else:
                ttk.Entry(controls, textvariable=self.vars[key], width=18).grid(row=row, column=1, sticky="ew", pady=2)
            row += 1
        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row += 1
        ttk.Label(controls, text="AC/CSP transient and structural tuning", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w"); row += 1
        for key, label, _ in AC_TUNING_FIELDS:
            self.vars[key] = tk.StringVar()
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=2)
            self._make_spinbox(controls, self.vars[key], key).grid(row=row, column=1, sticky="ew", pady=2)
            row += 1
        for prefix, title in (("lat", "Lateral Magic Formula"), ("lon", "Longitudinal Magic Formula")):
            ttk.Separator(controls).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row += 1
            ttk.Label(controls, text=title, font=("Segoe UI", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w"); row += 1
            for key, label, _ in AXIS_FIELDS:
                name = f"{prefix}_{key}"
                self.vars[name] = tk.StringVar()
                ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=2)
                self._make_spinbox(controls, self.vars[name], key).grid(row=row, column=1, sticky="ew", pady=2)
                row += 1

        self.figure = Figure(figsize=(9, 8), dpi=100, constrained_layout=True)
        self.axes = self.figure.subplots(3, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart)
        self.canvas.draw()
        toolbar = NavigationToolbar2Tk(self.canvas, chart, pack_toolbar=False)
        toolbar.update(); toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _make_spinbox(self, parent: ttk.Frame, variable: tk.StringVar, config_key: str) -> ttk.Spinbox:
        minimum, maximum, increment = SPIN_CONFIG[config_key]
        widget = ttk.Spinbox(
            parent, textvariable=variable, from_=minimum, to=maximum,
            increment=increment, width=18, command=self._on_numeric_change,
        )
        variable.trace_add("write", self._on_numeric_change)

        def wheel(event: tk.Event) -> str:
            widget.invoke("buttonup" if event.delta > 0 else "buttondown")
            return "break"

        widget.bind("<MouseWheel>", wheel)
        widget.bind("<KeyRelease>", self._on_numeric_change)
        return widget

    def _on_numeric_change(self, *_args: object) -> None:
        # A parameter edit invalidates the previous optimization result.
        self.fit = None
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(80, self._live_redraw)

    def _live_redraw(self) -> None:
        self._redraw_job = None
        self.redraw()

    def _set_defaults(self) -> None:
        for key, _, default in FIELDS:
            self.vars[key].set(default)
        for key, _, default in AC_TUNING_FIELDS:
            self.vars[key].set(default)
        defaults = {"lat": ["7.6494", "1.2348", "-1.2545", "2.4976", "-0.1500", "0", "0"],
                    "lon": ["6.4897", "1.2612", "-1.8430", "2.7111", "-0.1888", "0", "0"]}
        for prefix, values in defaults.items():
            for (key, _, _), value in zip(AXIS_FIELDS, values):
                self.vars[f"{prefix}_{key}"].set(value)

    def read_tyre(self) -> TyreDefinition:
        def axis(prefix: str) -> MagicFormulaAxis:
            return MagicFormulaAxis(**{key: float(self.vars[f"{prefix}_{key}"].get()) for key, _, _ in AXIS_FIELDS})
        return TyreDefinition(
            name=self.vars["name"].get().strip(), short_name=self.vars["short_name"].get().strip(),
            width_m=float(self.vars["width_m"].get()), radius_m=float(self.vars["radius_m"].get()),
            rim_radius_m=float(self.vars["rim_radius_m"].get()), reference_load_n=float(self.vars["reference_load_n"].get()),
            angular_inertia_kgm2=float(self.vars["angular_inertia_kgm2"].get()),
            tyre_damping_ns_m=float(self.vars["tyre_damping_ns_m"].get()),
            tyre_rate_n_m=float(self.vars["tyre_rate_n_m"].get()),
            pressure_static_psi=float(self.vars["pressure_static_psi"].get()), pressure_ideal_psi=float(self.vars["pressure_ideal_psi"].get()),
            relaxation_length_m=float(self.vars["relaxation_length_m"].get()),
            flex=float(self.vars["flex"].get()), flex_gain=float(self.vars["flex_gain"].get()),
            friction_limit_angle_deg=float(self.vars["friction_limit_angle_deg"].get()),
            lateral=axis("lat"), longitudinal=axis("lon"))

    def write_tyre(self, tyre: TyreDefinition) -> None:
        for key, _, _ in FIELDS:
            self.vars[key].set(str(getattr(tyre, key)))
        for key, _, _ in AC_TUNING_FIELDS:
            self.vars[key].set(str(getattr(tyre, key)))
        for prefix, axis in (("lat", tyre.lateral), ("lon", tyre.longitudinal)):
            for key, _, _ in AXIS_FIELDS:
                self.vars[f"{prefix}_{key}"].set(str(getattr(axis, key)))
        self.fit = None
        self.redraw()

    def redraw(self) -> None:
        try:
            tyre = self.read_tyre()
            # Use a fixed low-load test range so every preview is directly
            # comparable, regardless of the selected reference load FZ0.
            loads = np.array([200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0])
            configs = [("lateral", tyre.lateral, self.axes[0, 0], "Slip angle (deg)", "Lateral force Fy (N)"),
                       ("longitudinal", tyre.longitudinal, self.axes[0, 1], "Slip ratio", "Longitudinal force Fx (N)")]
            for axis_name, mf, ax, xlabel, ylabel in configs:
                ax.clear(); grid = slip_grid(axis_name)  # type: ignore[arg-type]
                x = np.rad2deg(grid) if axis_name == "lateral" else grid
                for fz in loads:
                    ax.plot(x, magic_formula(grid, fz, tyre.reference_load_n, mf), label=f"MF {fz:.0f}N")
                    if self.fit:
                        csp = self.fit.lateral if axis_name == "lateral" else self.fit.longitudinal
                        ax.plot(x, csp_force(grid, fz, tyre.reference_load_n, csp, axis_name), "--", alpha=.8, label=f"AC export {fz:.0f}N")
                ax.set_title("Lateral pure slip" if axis_name == "lateral" else "Longitudinal pure slip")
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=.3); ax.legend(fontsize=7, ncol=2)

            ax = self.axes[1, 0]; ax.clear()
            loads2 = np.linspace(200.0, 1200.0, 120)
            ratios = loads2 / tyre.reference_load_n
            for mf, label in ((tyre.lateral, "MF lateral"), (tyre.longitudinal, "MF longitudinal")):
                ax.plot(loads2, mf.mu0 * ratios ** mf.load_exp, label=label)
            if self.fit:
                for p, label in ((self.fit.lateral, "CSP lateral"), (self.fit.longitudinal, "CSP longitudinal")):
                    ax.plot(loads2, np.maximum(.05, p.mu0 + p.mu1 * (ratios - 1)) * ratios ** (p.load_exp - 1), "--", label=label)
            ax.set_title("Peak friction coefficient vs load"); ax.set_xlabel("Fz (N)"); ax.set_ylabel("Mu"); ax.grid(True, alpha=.3); ax.legend(fontsize=8)

            ax = self.axes[1, 1]; ax.clear()
            stiffness_loads = np.linspace(200.0, 1200.0, 120)
            mf_stiffness = np.array([
                cornering_stiffness_mf(fz, tyre) for fz in stiffness_loads
            ]) / 1000.0
            ax.plot(stiffness_loads / 1000.0, mf_stiffness, label="MF target")
            if self.fit:
                csp_stiffness = np.array([
                    cornering_stiffness_csp(fz, tyre.reference_load_n, self.fit.lateral)
                    for fz in stiffness_loads
                ]) / 1000.0
                ax.plot(stiffness_loads / 1000.0, csp_stiffness, "--", label="AC export")
            ax.axvline(tyre.reference_load_n / 1000.0, color="black", lw=.7,
                       alpha=.45, label="FZ0")
            ax.set_title("Cornering stiffness vs vertical load")
            ax.set_xlabel("Fz (kN)"); ax.set_ylabel("C-alpha (kN/rad)")
            ax.grid(True, alpha=.3); ax.legend(fontsize=8)

            fz_ref = tyre.reference_load_n
            ax = self.axes[2, 0]; ax.clear()
            if self.fit is None:
                ax.text(0.5, 0.5, "Click Fit CSP to show the friction circle",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title("Friction circle (combined slip)")
            else:
                for mult in (0.6, 1.4):
                    fx_e, fy_e = friction_ellipse(fz_ref * mult, fz_ref, self.fit)
                    ax.plot(fx_e, fy_e, color="0.55", lw=1.0, ls="--")
                fx_e, fy_e = friction_ellipse(fz_ref, fz_ref, self.fit)
                ax.plot(fx_e, fy_e, color="black", lw=1.6, ls="--", label="Ideal ellipse")
                kg, ag = np.meshgrid(np.linspace(-0.35, 0.35, 101),
                                     np.deg2rad(np.linspace(-18, 18, 101)))
                fxg, fyg = combined_force(kg.ravel(), ag.ravel(), fz_ref, fz_ref, self.fit, tyre)
                ax.scatter(fxg, fyg, s=2, alpha=.22, color="#2ca02c", label="Model envelope")
                ax.axhline(0, color="k", lw=.5, alpha=.4); ax.axvline(0, color="k", lw=.5, alpha=.4)
                ax.set_title(f"Friction circle @ {fz_ref:.0f} N (CF={tyre.combined_factor:.2f})")
                ax.set_xlabel("Fx (N)"); ax.set_ylabel("Fy (N)")
                ax.legend(fontsize=7); ax.grid(True, alpha=.3)
                ax.set_aspect("equal", adjustable="box")

            ax = self.axes[2, 1]; ax.clear()
            if self.fit is None:
                ax.text(0.5, 0.5, "Click Fit CSP to show combined-slip curves",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title("Combined slip: Fy vs alpha")
            else:
                alpha_c = slip_grid("lateral")
                for kappa_v in (0.0, 0.1, 0.2):
                    _, fy_c = combined_force(np.full_like(alpha_c, kappa_v), alpha_c,
                                             fz_ref, fz_ref, self.fit, tyre)
                    ax.plot(np.rad2deg(alpha_c), fy_c, lw=1.6, label=f"kappa = {kappa_v:.1f}")
                ax.set_title("Combined slip: Fy vs slip angle")
                ax.set_xlabel("Slip angle (deg)"); ax.set_ylabel("Fy (N)")
                ax.legend(fontsize=8); ax.grid(True, alpha=.3)

            self.canvas.draw_idle()
        except Exception as exc:
            self.status.set(f"Invalid parameter: {exc}")

    def run_fit(self) -> None:
        try:
            self.status.set("Fitting CSP parameters...")
            self.update_idletasks()
            self.fit = fit_csp(self.read_tyre())
            self.status.set(f"Fit complete: lateral R2 {self.fit.lateral.r_squared:.5f}, longitudinal R2 {self.fit.longitudinal.r_squared:.5f}")
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Fit failed", f"{exc}\n\n{traceback.format_exc()}")

    def save_design(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Tyre design", "*.json")])
        if path:
            save_definition(Path(path), self.read_tyre(), self.fit)
            self.status.set(f"Saved: {path}")

    def open_design(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Tyre design", "*.json")])
        if path:
            self.write_tyre(load_definition(Path(path)))
            self.status.set(f"Opened: {path}")

    def import_measured(self) -> None:
        path = filedialog.askopenfilename(
            title="Import measured UniTire model",
            filetypes=[("UniTire JSON model", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            tyre = load_unitire_definition(Path(path))
            self.write_tyre(tyre)
            self.status.set(
                f"Measured model imported: FZ0 {tyre.reference_load_n:.0f} N, "
                f"vertical rate {tyre.tyre_rate_n_m / 1000:.1f} kN/m, "
                f"relaxation {tyre.relaxation_length_m:.3f} m"
            )
        except Exception as exc:
            messagebox.showerror(
                "Import failed",
                f"The measured model could not be imported.\n\n{exc}\n\n{traceback.format_exc()}",
            )

    def export_ac(self) -> None:
        try:
            tyre = self.read_tyre()
            if self.fit is None:
                self.status.set("Fitting CSP parameters before export...")
                self.update_idletasks()
                self.fit = fit_csp(tyre)
                self.redraw()

            filename = filedialog.asksaveasfilename(
                title="Export Assetto Corsa tyre file",
                defaultextension=".ini",
                initialfile="tyres.ini",
                filetypes=[("Assetto Corsa tyre file", "*.ini"), ("All files", "*.*")],
            )
            if not filename:
                self.status.set("Export cancelled.")
                return

            selected = Path(filename)
            backup: Path | None = None
            if selected.exists():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = selected.with_name(f"{selected.name}.backup_{stamp}")
                shutil.copy2(selected, backup)
            generated = export_ac_package(selected.parent, tyre, self.fit, selected.name)
            file_list = "\n".join(str(path) for path in generated)
            backup_note = f"\n\nPrevious tyre file backed up to:\n{backup}" if backup else ""
            messagebox.showinfo(
                "Export complete",
                f"Generated and verified:\n\n{file_list}{backup_note}\n\n"
                "Back up the vehicle data folder and validate the result in-game.",
            )
            self.status.set(f"Exported and verified: {selected}")
        except Exception as exc:
            self.status.set(f"Export failed: {exc}")
            messagebox.showerror(
                "Export failed",
                f"The tyre package could not be written.\n\nError: {exc}\n\n"
                "Choose a writable folder outside Program Files, or run Content Manager "
                "with suitable permissions.\n\nTechnical details:\n{traceback.format_exc()}",
            )


def main() -> None:
    DesignerApp().mainloop()


if __name__ == "__main__":
    main()
