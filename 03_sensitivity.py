"""Etapa 3 — Análise de sensibilidade local (One-At-a-Time) de Ss e Sy.

Para cada perturbação ``p`` em ``config.SENSITIVITY_PERTURBATIONS`` e cada
parâmetro (Ss, Sy), executa o modelo, calcula a variação relativa nos heads
simulados e o índice de sensibilidade adimensional:

    S = (Δh / h) / (Δp / p)

onde Δh é a média absoluta da variação dos heads simulados em relação ao
baseline. Salva CSV em ``outputs/sensitivity/sensitivity.csv`` e gera
gráfico tornado em ``outputs/figures/tornado.png``.

Paralelização opcional (``config.SENSITIVITY_PARALLEL``) com
multiprocessing.Pool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from core import load_observations, run_model, setup_logging


@dataclass(frozen=True)
class PerturbCase:
    """Caso de perturbação OAT."""

    parameter: str  # "ss" ou "sy"
    perturbation: float  # ex.: -0.10


def _run_case(case: PerturbCase) -> dict:
    """Executa uma única perturbação e retorna heads + metadata."""
    ss = config.SS_INITIAL
    sy = config.SY_INITIAL
    if case.parameter == "ss":
        ss = config.SS_INITIAL * (1.0 + case.perturbation)
    else:
        sy = config.SY_INITIAL * (1.0 + case.perturbation)
    obs = load_observations()
    sim, _, ok = run_model(ss, sy, obs=obs)
    return {
        "parameter": case.parameter,
        "perturbation": case.perturbation,
        "ss": ss,
        "sy": sy,
        "converged": ok,
        "sim": sim,
    }


def _sensitivity_index(
    h_base: np.ndarray, h_pert: np.ndarray, p_pert: float,
) -> float:
    """S = (Δh̄ / h̄) / (Δp / p), com h̄ = média de |h_base|.

    Δh̄ é a média absoluta de (h_pert - h_base) sobre pontos válidos.
    """
    mask = np.isfinite(h_base) & np.isfinite(h_pert)
    if mask.sum() == 0 or p_pert == 0:
        return float("nan")
    dh = np.mean(np.abs(h_pert[mask] - h_base[mask]))
    h_ref = np.mean(np.abs(h_base[mask]))
    if h_ref == 0:
        return float("nan")
    return float((dh / h_ref) / abs(p_pert))


def _tornado_plot(df: pd.DataFrame, out_png: Path) -> None:
    """Gera gráfico tornado a partir do DataFrame de sensibilidade."""
    fig, ax = plt.subplots(figsize=(8, 5))
    params = sorted(df["parameter"].unique())
    perts = sorted(df["perturbation"].unique())
    width = 0.8 / len(perts)
    for i, p in enumerate(perts):
        sub = df[df["perturbation"] == p].set_index("parameter").reindex(params)
        ax.barh(
            np.arange(len(params)) + i * width,
            sub["sensitivity_index"].values,
            height=width,
            label=f"{p:+.0%}",
        )
    ax.set_yticks(np.arange(len(params)) + 0.4)
    ax.set_yticklabels(params)
    ax.set_xlabel("Índice de sensibilidade adimensional |S|")
    ax.set_title("Tornado — sensibilidade local (OAT) de Ss e Sy")
    ax.axvline(0, color="k", lw=0.5)
    ax.legend(title="Perturbação", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> pd.DataFrame:
    """Executa a sensibilidade completa e devolve DataFrame de resultados."""
    config.ensure_dirs()
    logger = setup_logging("03_sensitivity")
    logger.setLevel(logging.INFO)

    obs = load_observations()

    logger.info("Rodando baseline para referência")
    h_base, _, ok = run_model(config.SS_INITIAL, config.SY_INITIAL, obs=obs)
    if not ok:
        raise RuntimeError("Baseline não convergiu — abortando sensibilidade.")

    cases = [
        PerturbCase(parameter=p, perturbation=d)
        for p in ("ss", "sy")
        for d in config.SENSITIVITY_PERTURBATIONS
    ]
    logger.info("Total de casos OAT: %d", len(cases))

    if config.SENSITIVITY_PARALLEL:
        with Pool(config.SENSITIVITY_N_WORKERS) as pool:
            results = pool.map(_run_case, cases)
    else:
        results = [_run_case(c) for c in cases]

    rows = []
    for r in results:
        s = _sensitivity_index(h_base, r["sim"], r["perturbation"])
        rows.append({
            "parameter": r["parameter"],
            "perturbation": r["perturbation"],
            "ss": r["ss"],
            "sy": r["sy"],
            "converged": r["converged"],
            "sensitivity_index": abs(s) if np.isfinite(s) else np.nan,
        })
    df = pd.DataFrame(rows)
    out_csv = config.SENSITIVITY_DIR / "sensitivity.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Sensibilidade salva em %s", out_csv)

    out_png = config.FIGURES_DIR / "tornado.png"
    _tornado_plot(df, out_png)
    logger.info("Gráfico tornado em %s", out_png)
    return df


if __name__ == "__main__":
    main()
