"""Etapa 5 — Diagnósticos e relatório final.

Lê ``outputs/calibration/calibrated_heads.csv`` e gera:
  * Scatter observado vs. simulado com linha 1:1
  * Hidrogramas (sim × obs) por poço
  * Histograma de resíduos
  * Mapa espacial de resíduos médios
  * Métricas: RMSE, MAE, NSE, R², bias

Ao final, escreve ``outputs/calibration_report.md`` com interpretação
hidrogeológica dos parâmetros calibrados.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from core import compute_metrics, setup_logging


def _scatter(df: pd.DataFrame, out: Path) -> None:
    """Scatter observado vs simulado com linha 1:1."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["head_obs"], df["head_sim"], s=12, alpha=0.6)
    lo = float(min(df["head_obs"].min(), df["head_sim"].min()))
    hi = float(max(df["head_obs"].max(), df["head_sim"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Head observado [m]")
    ax.set_ylabel("Head simulado [m]")
    ax.set_title("Observado vs. simulado")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _hydrographs(df: pd.DataFrame, out_dir: Path) -> None:
    """Um hidrograma por poço (sim e obs vs. tempo)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for well, sub in df.groupby("well_id"):
        sub = sub.sort_values("date")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sub["date"], sub["head_obs"], "o-", label="Observado", ms=4)
        ax.plot(sub["date"], sub["head_sim"], "s--", label="Simulado", ms=4)
        ax.set_title(f"Poço {well}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Head [m]")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / f"hydrograph_{well}.png", dpi=130)
        plt.close(fig)


def _residual_hist(df: pd.DataFrame, out: Path) -> None:
    """Histograma de resíduos."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["residual"].dropna(), bins=30, edgecolor="k")
    ax.axvline(0, color="r", lw=1)
    ax.set_xlabel("Resíduo (sim - obs) [m]")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição dos resíduos")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _spatial_residuals(df: pd.DataFrame, out: Path) -> None:
    """Mapa de resíduos médios por poço."""
    by_well = df.groupby("well_id").agg(
        x=("x", "mean"), y=("y", "mean"), residual=("residual", "mean"),
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        by_well["x"], by_well["y"], c=by_well["residual"],
        cmap="RdBu_r", s=80, edgecolor="k",
        vmin=-by_well["residual"].abs().max(),
        vmax=by_well["residual"].abs().max(),
    )
    plt.colorbar(sc, ax=ax, label="Resíduo médio [m]")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Mapa espacial de resíduos médios")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _interpret(ss: float, sy: float) -> str:
    """Interpretação hidrogeológica simples dos valores calibrados."""
    parts = []
    if ss < 1e-5:
        parts.append("Ss baixo → aquífero confinado/semiconfinado pouco compressível.")
    elif ss < 1e-4:
        parts.append("Ss intermediário → comportamento típico de aquíferos confinados.")
    else:
        parts.append("Ss alto → aquífero pouco rígido ou parcialmente livre.")
    if sy < 0.05:
        parts.append("Sy baixo → matriz pouco drenável (silte/argila ou rocha fraturada).")
    elif sy < 0.20:
        parts.append("Sy intermediário → arenitos finos/sedimentos heterogêneos.")
    else:
        parts.append("Sy alto → areia/cascalho bem selecionado, drenagem eficiente.")
    return " ".join(parts)


def main() -> dict:
    """Gera figuras e relatório consolidado."""
    config.ensure_dirs()
    logger = setup_logging("05_diagnostics")

    csv = config.CALIBRATION_DIR / "calibrated_heads.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Rode 04_calibration.py primeiro. {csv} ausente.")
    df = pd.read_csv(csv, parse_dates=["date"])

    metrics = compute_metrics(df["head_obs"].to_numpy(), df["head_sim"].to_numpy())
    logger.info("Métricas: %s", metrics)

    _scatter(df, config.FIGURES_DIR / "scatter_obs_vs_sim.png")
    _hydrographs(df, config.FIGURES_DIR / "hydrographs")
    _residual_hist(df, config.FIGURES_DIR / "residual_histogram.png")
    _spatial_residuals(df, config.FIGURES_DIR / "spatial_residuals.png")

    res_json = config.CALIBRATION_DIR / "calibration_result.json"
    calib = json.loads(res_json.read_text(encoding="utf-8")) if res_json.exists() else {}

    ss = float(calib.get("ss", np.nan))
    sy = float(calib.get("sy", np.nan))
    cond = calib.get("condition_number")

    report = config.OUTPUT_DIR / "calibration_report.md"
    lines = [
        "# Relatório de calibração — Ss e Sy (transiente)",
        "",
        f"- Modelo: `{config.MODEL_DIR}`",
        f"- Versão MODFLOW: `{config.MODFLOW_VERSION}` | pacote: `{config.STORAGE_PACKAGE}`",
        f"- Período de calibração: {config.CALIB_START} → {config.CALIB_END}",
        "",
        "## Parâmetros calibrados",
        "",
        f"- **Ss** = `{ss:.3e}` [1/m]",
        f"- **Sy** = `{sy:.4f}` [-]",
        f"- Número de condição da Jacobiana: `{cond}`",
        "",
        "## Métricas finais",
        "",
        f"- RMSE: `{metrics['rmse']:.4f}` m",
        f"- MAE:  `{metrics['mae']:.4f}` m",
        f"- NSE:  `{metrics['nse']:.4f}`",
        f"- R²:   `{metrics['r2']:.4f}`",
        f"- bias: `{metrics['bias']:.4f}` m",
        f"- n:    `{metrics['n']}`",
        "",
        "## Interpretação hidrogeológica",
        "",
        _interpret(ss, sy),
        "",
        "## Figuras",
        "",
        "- `figures/scatter_obs_vs_sim.png`",
        "- `figures/residual_histogram.png`",
        "- `figures/spatial_residuals.png`",
        "- `figures/hydrographs/*.png`",
        "- `figures/tornado.png` (sensibilidade)",
        "",
    ]
    if cond is not None and cond > 1e6:
        lines.insert(
            -1,
            "> **Aviso**: número de condição alto — Ss e Sy podem ser pouco "
            "identificáveis. Considere mais observações de poços, períodos de "
            "bombeamento variáveis, ou parametrização zonal mais grosseira.",
        )
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Relatório em %s", report)
    return metrics


if __name__ == "__main__":
    main()
