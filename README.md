# CAST Benchmark Pipeline

Minimal pipeline for benchmarking spatial transcriptomics alignment using the CAST framework.

This script runs CAST_MARK + CAST_STACK on paired datasets and outputs aligned coordinates and QC metrics.

---


## Input Format

Each dataset pair requires:

### Metadata CSV

* `cell_id`
* `center_x`
* `center_y`

### Expression CSV

* rows: cell_id
* columns: genes

---

## Environment Variables

Required:

```
export CAST_DATA_DIR=/path/to/data
export CAST_OUT_DIR=/path/to/output

export CAST_PAIRS="meta_ref.csv,meta_mov.csv,expr_ref.csv,expr_mov.csv"
```

Optional:

```
export CAST_XCOL=center_x
export CAST_YCOL=center_y
export CAST_N_TOP_GENES=2000
```

---

## Run

```
python run_cast_align.py
```

---

## Output

For each dataset pair:

* `*_aligned.csv`
  Aligned coordinates of moving slice

* `*_qc.png`
  Before / after alignment visualization

* `*_shift_stats.csv`
  Displacement statistics

---

## Key Parameters

Defined inside `run_cast_align.py`:

* `iterations` — affine optimization steps
* `iterations_bs` — FFD deformation steps
* `graph_strategy` — "delaunay" or "knn"
* `bleeding` — spatial neighborhood for matching

Recommended settings for large datasets:

```
iterations = 200
iterations_bs = 100–200
graph_strategy = "knn"
```

---

## Purpose

This script is intended for:

* benchmarking CAST alignment
* comparing with other methods 
* rapid testing on new datasets

---
