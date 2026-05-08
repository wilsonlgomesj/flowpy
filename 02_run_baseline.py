"""Etapa 2 — Executa o modelo original e extrai heads simulados.

Roda uma única vez com (Ss, Sy) iniciais (do ``config``) em um diretório
isolado, extrai heads simulados nos pontos de observação e salva em
``outputs/baseline_heads.csv``. NÃO sobrescreve o modelo original.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

import config
from core import compute_metrics, load_observations, run_model, setup_logging


def main() -> pd.DataFrame:
    """Executa baseline e devolve DataFrame obs+sim."""
    config.ensure_dirs()
    logger = setup_logging("02_run_baseline")

    obs = load_observations()
    logger.info("Observações carregadas: %d pontos / %d poços",
                len(obs), obs["well_id"].nunique())

    logger.info("Rodando baseline com Ss=%.3e Sy=%.3f",
                config.SS_INITIAL, config.SY_INITIAL)
    sim, _, ok = run_model(config.SS_INITIAL, config.SY_INITIAL, obs=obs)
    if not ok:
        logger.error("Baseline não convergiu — verifique inputs do modelo.")

    df = obs.copy()
    df["head_sim"] = sim
    df["residual"] = df["head_sim"] - df["head_obs"]

    metrics = compute_metrics(df["head_obs"].to_numpy(), df["head_sim"].to_numpy())
    logger.info("Métricas baseline: %s", metrics)

    out_csv: Path = config.OUTPUT_DIR / "baseline_heads.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Heads salvos em %s", out_csv)

    with (config.OUTPUT_DIR / "baseline_metrics.pkl").open("wb") as fh:
        pickle.dump(metrics, fh)
    return df


if __name__ == "__main__":
    main()
