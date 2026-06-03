# Dataset Documentation

## Overview

This directory contains time-series measurements from a portable electronic nose (e-nose) device used to classify commercial hop varieties. Data was collected at CETENE (Centro de Tecnologias Estrategicas do Nordeste), Recife, Brazil.

The dataset consists of 30 consolidated CSV files organized by hop variety, conservation condition, and processing form. Total size is approximately 2.3 MB with around 30,600 measurement records.

## Sensor Array

The e-nose uses an array of 7 commercially available MOS (metal-oxide semiconductor) gas sensors from the TGS series by Figaro Engineering, plus 3 environmental sensors (temperature, atmospheric pressure, relative humidity). Refer to the paper in `paper/paper-hop.pdf` for detailed sensor specifications.

## Measurement Protocol

- **Aspiration phase:** 20s (micro-pump draws headspace air from sample)
- **Reading phase:** ~23s (sensor responses recorded)
- **Purge phase:** 60s (restore baseline conditions)
- **Cycles per sample:** 5
- **Total time per sample:** ~8.5 minutes
- **Inter-sample purge:** 120s
- **Acquisition rate:** ~3 Hz via serial communication

## CSV File Format

Files use tab-separated values (TSV) with 13 columns:

| Column | Description |
|--------|-------------|
| 0 | Cycle ID |
| 1 | Pump state |
| timestamp | Unix timestamp |
| TGS826 | Sensor reading (scientific notation) |
| TGS2611 | Sensor reading (scientific notation) |
| TGS2603 | Sensor reading (scientific notation) |
| TGS813 | Sensor reading (scientific notation) |
| TGS822 | Sensor reading (scientific notation) |
| TGS2602 | Sensor reading (scientific notation) |
| TGS823 | Sensor reading (scientific notation) |
| temperature | Temperature |
| pressao | Atmospheric pressure |
| humidity | Relative humidity |

Cycle boundaries are marked by `_CICLO_INICIADO_X_DE_Y_` and `_CICLO_FINALIZADO_` markers. Pump activation is marked by `_BOMBA_LIGADA_`.

## Dataset Categories

### Category A: Cone Varieties (whole hop flowers, 9 files)

These files contain measurements from hop cones (whole flowers) in various conservation states. The 9 cone files are those used in the IJCNN 2026 paper, representing 5 hop varieties across 9 distinct classes that capture both variety and storage condition. Total: 130 measurement cycles.

| File | Variety | Condition | Cycles | Size |
|------|---------|-----------|--------|------|
| chinook_fresco.csv | Chinook | Fresh | 15 | 126K |
| chinook_passada.csv | Chinook | Aged | 20 | 130K |
| come_2025_2.csv | Comet | 2025 vintage | 20 | 169K |
| saaz_2005.csv | Saaz | 2005 archive (20-year-old) | 15 | 170K |
| saaz_fresco.csv | Saaz | Fresh | 15 | 118K |
| saaz_passado.csv | Saaz | Aged | 10 | 79K |
| saaz_seco.csv | Saaz | Dry | 15 | 118K |
| vista_fresco.csv | Vista | Fresh | 10 | 79K |
| zeus_2025_2.csv | Zeus | 2025 vintage | 10 | 85K |

### Category B: Pelletized Hops (9 files)

Measurements from commercially pelletized hops. Pelletization compresses hop flowers into uniform pellets for storage and transport. These files cover 9 varieties (including Cascade, Nugget, Triple Pearl, and Triumph which are not present in the cone dataset). Each file contains 5 measurement cycles. Not used in the current paper but available for future work.

| File | Variety | Cycles | Size |
|------|---------|--------|------|
| cascade_pellete.csv | Cascade | 5 | 42K |
| chinook_pellete.csv | Chinook | 5 | 46K |
| comet_pellete.csv | Comet | 5 | 46K |
| nugget_pellete.csv | Nugget | 5 | 42K |
| saaz_pellete.csv | Saaz | 5 | 46K |
| triple_pearl_pellete.csv | Triple Pearl | 5 | 42K |
| triumph_pellete.csv | Triumph | 5 | 42K |
| vista_pellete.csv | Vista | 5 | 42K |
| zeus_pellete.csv | Zeus | 5 | 42K |

### Category C: Oil Extracts (12 files)

Measurements from hop oil extracts. Oil extraction isolates the essential oils (primarily myrcene, humulene, and caryophyllene) from hop material. Files with `_oe` suffix indicate oil extract. Most are derived from pelletized hops (`pelletizado_oe`), while two are from fresh cones (`fresco_oe`) and one from dry cones (`seco_oe`). Not used in the current paper.

| File | Variety | Source | Cycles | Size |
|------|---------|--------|--------|------|
| cascade_pelletizado_oe.csv | Cascade | Pellet | 5 | 42K |
| chinook_pelletizado_oe.csv | Chinook | Pellet | 5 | 45K |
| comet_fresco_oe.csv | Comet | Fresh cone | 5 | 42K |
| comet_pelletizado_oe.csv | Comet | Pellet | 5 | 42K |
| nugget_pelletizado_oe.csv | Nugget | Pellet | 5 | 42K |
| saaz_pelletizado_oe.csv | Saaz | Pellet | 5 | 42K |
| saaz_seco_oe.csv | Saaz | Dry cone | 5 | 42K |
| triple_pearl_pelletizado_oe.csv | Triple Pearl | Pellet | 5 | 46K |
| triumph_pelletizado_oe.csv | Triumph | Pellet | 5 | 42K |
| vista_pelletizado_oe.csv | Vista | Pellet | 5 | 42K |
| zeus_fresco_oe.csv | Zeus | Fresh cone | 5 | 45K |
| zeus_pelletizado_oe.csv | Zeus | Pellet | 5 | 42K |

## Consolidation Metadata

The file `consolidation_metadata.json` describes the source files merged into each consolidated CSV. Each consolidated file was created by merging 2-4 individual measurement sessions. Consolidation preserves original cycle markers and timestamps.

## Usage Notes

- Only the 9 cone variety files (Category A) are used in the published paper.
- Pellet and oil extract files (Categories B and C) are provided for future research.
- The `run_hop_experiments.py` script references the 9 cone files via the `CLASS_FILES` dictionary.
- To add new data categories, modify `CLASS_FILES` in `run_hop_experiments.py`.
