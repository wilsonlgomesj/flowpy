"""Configuração central do pipeline de sensibilidade e calibração transiente.

Todos os caminhos, parâmetros e opções devem ser ajustados aqui antes de
executar os scripts numerados. Os valores marcados como ``PREENCHER`` precisam
ser definidos pelo usuário de acordo com o modelo MODFLOW específico.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# 1. Caminhos do modelo
# ---------------------------------------------------------------------------

MODEL_DIR: Path = Path(r"C:\Users\wilso\Downloads\LHG\TRANSIENTE_OFICIAL\08052026")
NAM_FILE: str = "PREENCHER.nam"

MODFLOW_VERSION: Literal["mf6", "mf2005", "mfnwt"] = "mf6"
MODFLOW_EXE: Path = Path("mf6.exe")

STORAGE_PACKAGE: Literal["STO", "LPF", "UPW"] = "STO"

# ---------------------------------------------------------------------------
# 2. Diretórios de saída do pipeline
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
SENSITIVITY_DIR: Path = OUTPUT_DIR / "sensitivity"
CALIBRATION_DIR: Path = OUTPUT_DIR / "calibration"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
RUNS_DIR: Path = OUTPUT_DIR / "runs"
LOG_DIR: Path = OUTPUT_DIR / "logs"

# ---------------------------------------------------------------------------
# 3. Parametrização de Ss e Sy
# ---------------------------------------------------------------------------

PARAMETERIZATION: Literal["homogeneous", "by_layer", "zonal"] = "homogeneous"

ZONES_FILE: Path | None = None  # ex.: PROJECT_ROOT / "zones.npy"

SS_BOUNDS: tuple[float, float] = (1e-6, 1e-3)
SY_BOUNDS: tuple[float, float] = (1e-2, 0.35)

SS_INITIAL: float = 1e-5
SY_INITIAL: float = 0.15

# ---------------------------------------------------------------------------
# 4. Observações
# ---------------------------------------------------------------------------

OBS_FILE: Path = PROJECT_ROOT / "observations.csv"

OBS_COLUMNS = {
    "well_id": "poco_id",
    "date": "data",
    "head_obs": "head_observado",
    "x": "x",
    "y": "y",
    "layer": "camada",
}

CALIB_START: str = "2020-01-01"
CALIB_END: str = "2024-12-31"

MODEL_START_DATE: str = "2020-01-01"

# ---------------------------------------------------------------------------
# 5. Sensibilidade
# ---------------------------------------------------------------------------

SENSITIVITY_PERTURBATIONS: tuple[float, ...] = (-0.50, -0.25, -0.10, 0.10, 0.25, 0.50)
SENSITIVITY_PARALLEL: bool = False
SENSITIVITY_N_WORKERS: int = 4

# ---------------------------------------------------------------------------
# 6. Calibração
# ---------------------------------------------------------------------------

CALIB_METHOD: Literal["differential_evolution", "lbfgsb"] = "differential_evolution"
CALIB_MAXITER: int = 50
CALIB_POPSIZE: int = 15
CALIB_TOL: float = 1e-4
CALIB_SEED: int = 42

PENALTY_NON_CONVERGENCE: float = 1e10

# ---------------------------------------------------------------------------
# 7. Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = "INFO"
LOG_FILE: Path = LOG_DIR / "pipeline.log"


def ensure_dirs() -> None:
    """Cria todos os diretórios de saída se não existirem."""
    for d in (OUTPUT_DIR, SENSITIVITY_DIR, CALIBRATION_DIR, FIGURES_DIR, RUNS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
