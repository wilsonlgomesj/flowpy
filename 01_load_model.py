"""Etapa 1 — Carrega o modelo MODFLOW e valida transiência.

Imprime resumo dimensional/temporal e checa se há ao menos um stress period
em regime transiente. Salva o resumo em ``outputs/model_summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from core import is_transient, load_model, model_summary, setup_logging


def main() -> dict:
    """Carrega o modelo, valida transiência e devolve o resumo."""
    config.ensure_dirs()
    logger = setup_logging("01_load_model")
    logger.info("Carregando modelo de %s", config.MODEL_DIR)

    handles = load_model()
    summary = model_summary(handles)
    logger.info(
        "Resumo: nlay=%s nrow=%s ncol=%s nper=%s",
        summary["nlay"], summary["nrow"], summary["ncol"], summary["nper"],
    )
    logger.info("perlen=%s", summary["perlen"])
    logger.info("nstp=%s", summary["nstp"])

    transient = is_transient(handles)
    summary["transient"] = transient
    if not transient:
        logger.warning("ATENÇÃO: nenhum stress period em regime transiente.")
    else:
        logger.info("Modelo em regime transiente: OK.")

    out_path: Path = config.OUTPUT_DIR / "model_summary.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str, ensure_ascii=False)
    logger.info("Resumo salvo em %s", out_path)
    return summary


if __name__ == "__main__":
    main()
