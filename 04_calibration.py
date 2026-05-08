"""Etapa 4 — Calibração de Ss e Sy contra heads observados.

Trabalha em log10(Ss) e log10(Sy) para melhorar o condicionamento numérico.
Usa ``scipy.optimize.differential_evolution`` (global) por padrão ou
``minimize`` L-BFGS-B (local) conforme ``config.CALIB_METHOD``. Também
calcula o número de condição da Jacobiana finite-difference para alertar
quando os parâmetros são pouco identificáveis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

import config
from core import (
    compute_metrics,
    load_observations,
    rmse_objective,
    run_model,
    setup_logging,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Resultado consolidado da calibração."""

    ss: float
    sy: float
    rmse: float
    metrics: dict
    n_evals: int
    success: bool
    condition_number: float | None
    history: list[dict]


def _bounds_log10() -> list[tuple[float, float]]:
    return [
        (float(np.log10(config.SS_BOUNDS[0])), float(np.log10(config.SS_BOUNDS[1]))),
        (float(np.log10(config.SY_BOUNDS[0])), float(np.log10(config.SY_BOUNDS[1]))),
    ]


def make_objective(obs_df: pd.DataFrame, history: list[dict]) -> Callable[[np.ndarray], float]:
    """Cria f([log10_ss, log10_sy]) -> RMSE com logging por avaliação."""
    obs_arr = obs_df["head_obs"].to_numpy()

    def f(x: np.ndarray) -> float:
        ss = float(10.0 ** x[0])
        sy = float(10.0 ** x[1])
        sim, _, ok = run_model(ss, sy, obs=obs_df)
        if not ok:
            rmse = config.PENALTY_NON_CONVERGENCE
        else:
            rmse = rmse_objective(obs_arr, sim)
        history.append({"ss": ss, "sy": sy, "rmse": rmse, "converged": ok})
        logger.info("eval #%d Ss=%.3e Sy=%.3f RMSE=%.4f", len(history), ss, sy, rmse)
        return rmse

    return f


def _jacobian_condition_number(
    obs_df: pd.DataFrame, ss: float, sy: float, eps: float = 1e-2,
) -> float | None:
    """Aproxima a Jacobiana ∂h/∂(log10 p) por diferenças finitas centrais.

    Retorna o número de condição (cond_2) da matriz J. Valores muito altos
    (> 1e6) indicam parâmetros pouco identificáveis.
    """
    try:
        sim_base, _, ok = run_model(ss, sy, obs=obs_df)
        if not ok:
            return None
        cols = []
        for which in ("ss", "sy"):
            base = ss if which == "ss" else sy
            p_plus = base * (1.0 + eps)
            p_minus = base * (1.0 - eps)
            if which == "ss":
                sp, _, _ = run_model(p_plus, sy, obs=obs_df)
                sm, _, _ = run_model(p_minus, sy, obs=obs_df)
            else:
                sp, _, _ = run_model(ss, p_plus, obs=obs_df)
                sm, _, _ = run_model(ss, p_minus, obs=obs_df)
            dlog = np.log10(p_plus) - np.log10(p_minus)
            col = (sp - sm) / dlog
            cols.append(col)
        J = np.column_stack(cols)
        mask = np.all(np.isfinite(J), axis=1)
        if mask.sum() < 2:
            return None
        return float(np.linalg.cond(J[mask]))
    except Exception as exc:
        logger.warning("Falha ao calcular condicionamento: %s", exc)
        return None


def main() -> CalibrationResult:
    """Executa calibração e salva resultados."""
    config.ensure_dirs()
    setup_logging("04_calibration")

    obs = load_observations()
    history: list[dict] = []
    objective = make_objective(obs, history)

    bounds = _bounds_log10()
    logger.info("Bounds em log10: %s", bounds)
    logger.info("Método: %s", config.CALIB_METHOD)

    if config.CALIB_METHOD == "differential_evolution":
        res = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=config.CALIB_MAXITER,
            popsize=config.CALIB_POPSIZE,
            tol=config.CALIB_TOL,
            seed=config.CALIB_SEED,
            polish=True,
            updating="deferred",
            workers=1,
        )
        x_opt = res.x
        success = bool(res.success)
        rmse_opt = float(res.fun)
    else:
        x0 = np.array([np.log10(config.SS_INITIAL), np.log10(config.SY_INITIAL)])
        res = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": config.CALIB_MAXITER, "ftol": config.CALIB_TOL},
        )
        x_opt = res.x
        success = bool(res.success)
        rmse_opt = float(res.fun)

    ss_opt = float(10.0 ** x_opt[0])
    sy_opt = float(10.0 ** x_opt[1])
    logger.info("Ótimo: Ss=%.3e Sy=%.4f RMSE=%.4f", ss_opt, sy_opt, rmse_opt)

    sim_opt, _, ok = run_model(ss_opt, sy_opt, obs=obs)
    metrics = compute_metrics(obs["head_obs"].to_numpy(), sim_opt) if ok else {}

    cond = _jacobian_condition_number(obs, ss_opt, sy_opt)
    if cond is not None and cond > 1e6:
        logger.warning(
            "Número de condição alto (%.2e): Ss e Sy podem ser pouco "
            "identificáveis com as observações disponíveis.", cond,
        )

    result = CalibrationResult(
        ss=ss_opt,
        sy=sy_opt,
        rmse=rmse_opt,
        metrics=metrics,
        n_evals=len(history),
        success=success,
        condition_number=cond,
        history=history,
    )

    out_json: Path = config.CALIBRATION_DIR / "calibration_result.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "ss": result.ss,
                "sy": result.sy,
                "rmse": result.rmse,
                "metrics": result.metrics,
                "n_evals": result.n_evals,
                "success": result.success,
                "condition_number": result.condition_number,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    pd.DataFrame(history).to_csv(config.CALIBRATION_DIR / "calibration_history.csv", index=False)

    if ok:
        df = obs.copy()
        df["head_sim"] = sim_opt
        df["residual"] = df["head_sim"] - df["head_obs"]
        df.to_csv(config.CALIBRATION_DIR / "calibrated_heads.csv", index=False)

    logger.info("Resultados em %s", config.CALIBRATION_DIR)
    return result


if __name__ == "__main__":
    main()
