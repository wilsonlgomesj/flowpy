# flowpy — Análise de Sensibilidade e Calibração Transiente (Ss, Sy)

Pipeline modular em Python/FloPy para executar um modelo MODFLOW transiente,
analisar a sensibilidade dos parâmetros de armazenamento e calibrar Ss e Sy
contra heads observados em poços. Compatível com **MODFLOW 6**, **MODFLOW-2005**
e **MODFLOW-NWT** (pacotes `STO`, `LPF`, `UPW`).

## Estrutura

```
flowpy/
├── config.py              # Caminhos, parâmetros e opções (EDITAR PRIMEIRO)
├── core.py                # Funções compartilhadas: load_model, run_model, métricas
├── 01_load_model.py       # Carrega e valida o modelo, salva resumo
├── 02_run_baseline.py     # Executa baseline e extrai heads simulados
├── 03_sensitivity.py      # Sensibilidade local OAT (±10/25/50%) + tornado
├── 04_calibration.py      # Calibração com differential_evolution ou L-BFGS-B
├── 05_diagnostics.py      # Scatter, hidrogramas, resíduos, mapa, relatório
├── requirements.txt
└── outputs/               # Criado automaticamente
    ├── runs/              # Diretórios isolados por rodada (descartáveis)
    ├── sensitivity/       # CSV de sensibilidade
    ├── calibration/       # JSON/CSV/histórico da calibração
    ├── figures/           # Todas as figuras (PNG)
    ├── logs/              # pipeline.log
    └── calibration_report.md
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```

## Configuração (obrigatório)

Edite `config.py`:

1. `MODEL_DIR` — caminho do modelo MODFLOW
2. `NAM_FILE` — apenas para MF2005/NWT
3. `MODFLOW_VERSION` — `"mf6"`, `"mf2005"` ou `"mfnwt"`
4. `MODFLOW_EXE` — caminho do executável (`mf6.exe`, `mf2005.exe`, …)
5. `STORAGE_PACKAGE` — `"STO"` (MF6), `"LPF"` ou `"UPW"`
6. `OBS_FILE` — CSV/Excel de observações com colunas
   `poco_id, data, head_observado, x, y, camada`
7. `MODEL_START_DATE`, `CALIB_START`, `CALIB_END` — janela temporal
8. (opcional) `PARAMETERIZATION = "zonal"` + `ZONES_FILE` apontando para
   um `.npy` com array de zonas

## Execução

Rode os scripts na ordem. Cada um pode ser chamado isoladamente:

```bash
python 01_load_model.py     # valida modelo
python 02_run_baseline.py   # roda baseline, extrai heads
python 03_sensitivity.py    # sensibilidade OAT + gráfico tornado
python 04_calibration.py    # calibra Ss e Sy
python 05_diagnostics.py    # figuras + relatório final
```

> O modelo original em `MODEL_DIR` **nunca** é sobrescrito. Cada rodada
> ocorre em uma subpasta única em `outputs/runs/`.

## Notas técnicas

- Trabalho em `log10(Ss)` e `log10(Sy)` melhora o condicionamento numérico
  e dá pesos comparáveis às duas ordens de grandeza distintas.
- `Ss` afeta a resposta apenas em camadas confinadas; `Sy`, em camadas
  livres (`laytyp`/`iconvert > 0`). O código aplica ambos sem distinção,
  mas só os valores fisicamente relevantes têm efeito.
- Se o número de condição da Jacobiana ficar acima de `1e6`, o relatório
  emite aviso de baixa identificabilidade — adicione observações ou
  agrupe zonas.
- Modelos não convergentes recebem `RMSE = 1e10` para não quebrar o
  otimizador.
- Paralelização da sensibilidade é opcional (`SENSITIVITY_PARALLEL = True`).

## Logs

Tudo é logado em `outputs/logs/pipeline.log` e no console.
