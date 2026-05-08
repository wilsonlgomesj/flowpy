"""Módulo central com funções compartilhadas pelos scripts do pipeline.

Concentra carregamento do modelo, modificação do pacote de armazenamento,
execução em diretório temporário e extração de heads simulados nos pontos
de observação. Todos os scripts numerados (01_..05_) reutilizam estas funções.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import flopy
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str | None = None) -> logging.Logger:
    """Configura logging para console + arquivo. Idempotente."""
    config.ensure_dirs()
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(getattr(logging, config.LOG_LEVEL))
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
        fh = logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return logging.getLogger(name or __name__)


# ---------------------------------------------------------------------------
# Carregamento do modelo
# ---------------------------------------------------------------------------

@dataclass
class ModelHandles:
    """Estruturas FloPy do modelo carregado."""

    sim: flopy.mf6.MFSimulation | None
    model: flopy.mf6.ModflowGwf | flopy.modflow.Modflow
    version: str
    workspace: Path


def load_model(workspace: Path | None = None) -> ModelHandles:
    """Carrega o modelo MODFLOW de um diretório.

    Parameters
    ----------
    workspace
        Diretório do modelo. Se ``None``, usa ``config.MODEL_DIR``.

    Returns
    -------
    ModelHandles
        Objeto com a simulação (apenas MF6) e o modelo principal.
    """
    workspace = Path(workspace) if workspace else config.MODEL_DIR
    if not workspace.exists():
        raise FileNotFoundError(f"Diretório do modelo não encontrado: {workspace}")

    if config.MODFLOW_VERSION == "mf6":
        sim = flopy.mf6.MFSimulation.load(
            sim_ws=str(workspace),
            exe_name=str(config.MODFLOW_EXE),
        )
        model_names = sim.model_names
        if not model_names:
            raise RuntimeError("Simulação MF6 não contém modelos.")
        model = sim.get_model(model_names[0])
        return ModelHandles(sim=sim, model=model, version="mf6", workspace=workspace)

    # MF2005 / NWT
    ml = flopy.modflow.Modflow.load(
        config.NAM_FILE,
        model_ws=str(workspace),
        exe_name=str(config.MODFLOW_EXE),
        version=config.MODFLOW_VERSION,
        check=False,
    )
    return ModelHandles(sim=None, model=ml, version=config.MODFLOW_VERSION, workspace=workspace)


def is_transient(handles: ModelHandles) -> bool:
    """Retorna True se ao menos um stress period for transiente."""
    if handles.version == "mf6":
        tdis = handles.sim.tdis
        # MF6 não marca steady por SP no TDIS; a transiência fica no STO.
        sto = handles.model.get_package("sto")
        if sto is None:
            return False
        try:
            transient = sto.transient.get_data()
        except Exception:
            return True  # se há STO, assumimos transiência
        return any(bool(v) for v in (transient or {}).values())
    # MF2005/NWT: dis.steady é uma lista booleana por SP
    dis = handles.model.dis
    return any(not bool(s) for s in dis.steady.array)


def model_summary(handles: ModelHandles) -> dict:
    """Resumo dimensional e temporal do modelo."""
    m = handles.model
    if handles.version == "mf6":
        dis = m.dis
        tdis = handles.sim.tdis
        nper = tdis.nper.get_data()
        perlen = [p[0] for p in tdis.perioddata.get_data()]
        nstp = [p[1] for p in tdis.perioddata.get_data()]
        nlay = dis.nlay.get_data()
        nrow = dis.nrow.get_data() if hasattr(dis, "nrow") else None
        ncol = dis.ncol.get_data() if hasattr(dis, "ncol") else None
    else:
        dis = m.dis
        nper, nlay, nrow, ncol = dis.nper, dis.nlay, dis.nrow, dis.ncol
        perlen = list(dis.perlen.array)
        nstp = list(dis.nstp.array)
    return {
        "nlay": int(nlay),
        "nrow": int(nrow) if nrow is not None else None,
        "ncol": int(ncol) if ncol is not None else None,
        "nper": int(nper),
        "perlen": perlen,
        "nstp": nstp,
    }


# ---------------------------------------------------------------------------
# Diretórios de execução isolados
# ---------------------------------------------------------------------------

def make_run_workspace(prefix: str = "run_") -> Path:
    """Cria diretório de execução isolado dentro de ``RUNS_DIR``.

    Cada chamada cria uma pasta única — nunca sobrescreve o modelo original.
    """
    config.ensure_dirs()
    return Path(tempfile.mkdtemp(prefix=prefix, dir=config.RUNS_DIR))


def copy_model_to(workspace: Path) -> Path:
    """Copia o modelo base para o diretório de trabalho."""
    workspace.mkdir(parents=True, exist_ok=True)
    for item in config.MODEL_DIR.iterdir():
        dst = workspace / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    return workspace


# ---------------------------------------------------------------------------
# Modificação do pacote de armazenamento
# ---------------------------------------------------------------------------

def _broadcast(value: float | np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == shape:
        return arr
    return np.full(shape, float(arr))


def apply_storage_params(
    handles: ModelHandles,
    ss: float | np.ndarray,
    sy: float | np.ndarray,
    zones: np.ndarray | None = None,
    ss_by_zone: dict[int, float] | None = None,
    sy_by_zone: dict[int, float] | None = None,
) -> None:
    """Aplica Ss e Sy ao pacote de armazenamento do modelo.

    Suporta parametrização homogênea, por camada ou zonal. Ss só é
    significativo em camadas confinadas; Sy em camadas livres
    (laytyp/iconvert > 0). Mesmo assim, escrevemos ambos para evitar erros.
    """
    m = handles.model

    if handles.version == "mf6":
        sto = m.get_package("sto")
        if sto is None:
            raise RuntimeError("Modelo MF6 sem pacote STO.")
        nlay = m.dis.nlay.get_data()
        nrow = m.dis.nrow.get_data()
        ncol = m.dis.ncol.get_data()
        shape = (nlay, nrow, ncol)
        ss_arr = _build_param_array(ss, shape, zones, ss_by_zone)
        sy_arr = _build_param_array(sy, shape, zones, sy_by_zone)
        sto.ss.set_data(ss_arr)
        sto.sy.set_data(sy_arr)
        return

    # LPF / UPW
    pkg_name = config.STORAGE_PACKAGE.lower()
    pkg = m.get_package(pkg_name)
    if pkg is None:
        raise RuntimeError(f"Pacote {config.STORAGE_PACKAGE} não encontrado.")
    nlay, nrow, ncol = m.dis.nlay, m.dis.nrow, m.dis.ncol
    shape = (nlay, nrow, ncol)
    ss_arr = _build_param_array(ss, shape, zones, ss_by_zone)
    sy_arr = _build_param_array(sy, shape, zones, sy_by_zone)
    pkg.ss = ss_arr
    pkg.sy = sy_arr


def _build_param_array(
    value: float | np.ndarray,
    shape: tuple[int, int, int],
    zones: np.ndarray | None,
    by_zone: dict[int, float] | None,
) -> np.ndarray:
    """Constrói array (nlay, nrow, ncol) conforme a parametrização."""
    if zones is not None and by_zone is not None:
        out = np.full(shape, np.nan)
        zones_b = np.broadcast_to(zones, shape)
        for z, v in by_zone.items():
            out[zones_b == z] = float(v)
        if np.isnan(out).any():
            out[np.isnan(out)] = float(np.asarray(value).mean()) if np.ndim(value) else float(value)
        return out
    return _broadcast(value, shape)


# ---------------------------------------------------------------------------
# Execução do modelo
# ---------------------------------------------------------------------------

def write_and_run(handles: ModelHandles) -> bool:
    """Reescreve os inputs e executa o MODFLOW. Retorna True se convergiu."""
    if handles.version == "mf6":
        handles.sim.write_simulation(silent=True)
        success, _ = handles.sim.run_simulation(silent=True)
    else:
        handles.model.write_input()
        success, _ = handles.model.run_model(silent=True)
    return bool(success)


# ---------------------------------------------------------------------------
# Observações e extração de heads
# ---------------------------------------------------------------------------

def load_observations() -> pd.DataFrame:
    """Carrega observações do CSV/Excel e padroniza colunas."""
    path = config.OBS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de observações não encontrado: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    rename = {v: k for k, v in config.OBS_COLUMNS.items()}
    df = df.rename(columns=rename)
    required = {"well_id", "date", "head_obs", "x", "y", "layer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes nas observações: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["layer"] = df["layer"].astype(int)
    start = pd.to_datetime(config.CALIB_START)
    end = pd.to_datetime(config.CALIB_END)
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    return df


def _row_col_from_xy(handles: ModelHandles, x: float, y: float) -> tuple[int, int]:
    """Converte coordenada (x, y) em (row, col) usando o modelgrid."""
    grid = handles.model.modelgrid
    row, col = grid.intersect(x, y, forgive=True)
    return int(row), int(col)


def _build_time_index(handles: ModelHandles) -> pd.DatetimeIndex:
    """Constrói índice temporal (uma data por timestep) a partir do TDIS."""
    summary = model_summary(handles)
    start = pd.to_datetime(config.MODEL_START_DATE)
    times: list[pd.Timestamp] = []
    cumulative = 0.0
    for perlen, nstp in zip(summary["perlen"], summary["nstp"]):
        dt = float(perlen) / max(int(nstp), 1)
        for k in range(int(nstp)):
            cumulative_end = cumulative + dt * (k + 1)
            times.append(start + timedelta(days=cumulative_end))
        cumulative += float(perlen)
    return pd.DatetimeIndex(times)


def extract_heads_at_obs(
    handles: ModelHandles,
    obs: pd.DataFrame,
) -> np.ndarray:
    """Extrai heads simulados alinhados com cada observação.

    Retorna array 1D de mesmo comprimento de ``obs``. Posições não
    encontradas (fora do grid, sem head válido) recebem NaN.
    """
    ws = handles.workspace
    head_files = list(ws.glob("*.hds")) + list(ws.glob("*.bhd"))
    if not head_files:
        raise FileNotFoundError(f"Nenhum arquivo .hds em {ws}")
    hf = flopy.utils.HeadFile(str(head_files[0]))
    times_sim = _build_time_index(handles)
    kstpkper = hf.get_kstpkper()

    sim = np.full(len(obs), np.nan)
    cache: dict[tuple[int, int], np.ndarray] = {}
    for i, row in obs.iterrows():
        idx = int(np.argmin(np.abs((times_sim - row["date"]).total_seconds())))
        idx = min(idx, len(kstpkper) - 1)
        key = kstpkper[idx]
        if key not in cache:
            cache[key] = hf.get_data(kstpkper=key)
        head_3d = cache[key]
        try:
            r, c = _row_col_from_xy(handles, float(row["x"]), float(row["y"]))
        except Exception:
            continue
        layer = int(row["layer"])
        if 0 <= layer < head_3d.shape[0] and 0 <= r < head_3d.shape[1] and 0 <= c < head_3d.shape[2]:
            v = head_3d[layer, r, c]
            if v > -1e29:
                sim[i] = float(v)
    hf.close()
    return sim


# ---------------------------------------------------------------------------
# run_model: a função do enunciado
# ---------------------------------------------------------------------------

def run_model(
    ss: float,
    sy: float,
    obs: pd.DataFrame | None = None,
    keep_workspace: bool = False,
) -> tuple[np.ndarray, Path, bool]:
    """Executa o modelo com (Ss, Sy) e devolve heads simulados nos poços.

    Parameters
    ----------
    ss, sy
        Valores escalares (homogêneos) ou já difundidos via zonas.
    obs
        DataFrame de observações; se None, é carregado do disco.
    keep_workspace
        Se True, não remove o diretório temporário no fim (útil p/ debug).

    Returns
    -------
    sim_heads, workspace, converged
    """
    if obs is None:
        obs = load_observations()

    workspace = make_run_workspace()
    try:
        copy_model_to(workspace)
        handles = load_model(workspace)
        zones = np.load(config.ZONES_FILE) if config.ZONES_FILE else None
        apply_storage_params(handles, ss=ss, sy=sy, zones=zones)
        converged = write_and_run(handles)
        if not converged:
            logger.warning("Modelo não convergiu (Ss=%.3e, Sy=%.3e)", ss, sy)
            return np.full(len(obs), np.nan), workspace, False
        sim = extract_heads_at_obs(handles, obs)
        return sim, workspace, True
    finally:
        if not keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, NSE, R², bias — ignora NaN."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return {"rmse": np.nan, "mae": np.nan, "nse": np.nan, "r2": np.nan, "bias": np.nan, "n": int(mask.sum())}
    o, s = obs[mask], sim[mask]
    res = s - o
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    bias = float(np.mean(res))
    denom = float(np.sum((o - o.mean()) ** 2))
    nse = float(1.0 - np.sum(res ** 2) / denom) if denom > 0 else np.nan
    if denom > 0:
        r = np.corrcoef(o, s)[0, 1]
        r2 = float(r ** 2)
    else:
        r2 = np.nan
    return {"rmse": rmse, "mae": mae, "nse": nse, "r2": r2, "bias": bias, "n": int(mask.sum())}


def rmse_objective(obs_heads: np.ndarray, sim_heads: np.ndarray) -> float:
    """RMSE com penalidade para falta de pontos válidos."""
    mask = np.isfinite(obs_heads) & np.isfinite(sim_heads)
    if mask.sum() < 2:
        return config.PENALTY_NON_CONVERGENCE
    res = sim_heads[mask] - obs_heads[mask]
    return float(np.sqrt(np.mean(res ** 2)))
