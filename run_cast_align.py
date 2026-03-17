import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch
from scipy import sparse

from CAST import CAST_MARK, CAST_STACK
from CAST.CAST_Stack import reg_params

warnings.filterwarnings("ignore")


def get_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        raise ValueError(f"Missing env variable: {name}")
    return v


def parse_pairs(pairs_str: str):
    """
    CAST_PAIRS format:
    meta_ref,meta_mov,expr_ref,expr_mov;meta_ref2,meta_mov2,expr_ref2,expr_mov2
    """
    pairs = []
    for item in pairs_str.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(",")]
        if len(parts) != 4:
            raise ValueError(
                "CAST_PAIRS must contain 4 comma-separated fields per pair: "
                "meta_ref,meta_mov,expr_ref,expr_mov"
            )
        pairs.append(parts)
    return pairs


def ensure_dense_float32(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def check_coords(coords: np.ndarray, name: str):
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{name}: coords must be (n_cells, 2), got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError(f"{name}: no cells")
    if not np.isfinite(coords).all():
        raise ValueError(f"{name}: coords contain NaN/Inf")


def load_slice(
    data_dir: str,
    meta_file: str,
    expr_file: str,
    xcol: str,
    ycol: str,
    name: str,
):
    """
    Read metadata + expression csv and build AnnData.
    Important:
    - keep spatial coords in original scale
    - do NOT log1p before CAST_MARK
    """
    meta_path = os.path.join(data_dir, meta_file)
    expr_path = os.path.join(data_dir, expr_file)

    print(f"\nLoading {name} metadata: {meta_path}")
    print(f"Loading {name} expression: {expr_path}")

    meta = pd.read_csv(meta_path)
    expr = pd.read_csv(expr_path, index_col=0)

    if meta.shape[1] == 0:
        raise ValueError(f"{name}: empty metadata file")
    if expr.shape[0] == 0 or expr.shape[1] == 0:
        raise ValueError(f"{name}: empty expression file")

    # Normalize cell_id column
    if meta.columns[0] != "cell_id":
        meta = meta.rename(columns={meta.columns[0]: "cell_id"})
    meta["cell_id"] = meta["cell_id"].astype(str)
    meta = meta.set_index("cell_id")

    expr.index = expr.index.astype(str)

    common = meta.index.intersection(expr.index)
    if len(common) == 0:
        raise ValueError(f"{name}: no overlapping cell ids between metadata and expression")

    meta = meta.loc[common].copy()
    expr = expr.loc[common].copy()

    if xcol not in meta.columns or ycol not in meta.columns:
        raise ValueError(
            f"{name}: missing coordinate columns. Need {xcol} and {ycol}, "
            f"but metadata has: {list(meta.columns[:20])}"
        )

    coords = meta[[xcol, ycol]].to_numpy(dtype=np.float32)
    check_coords(coords, name)

    X = expr.to_numpy(dtype=np.float32)
    adata = sc.AnnData(X=X)
    adata.obs_names = expr.index.astype(str)
    adata.var_names = expr.columns.astype(str)
    adata.obs = meta.copy()
    adata.obsm["spatial"] = coords

    # For CAST_MARK, keep values closer to counts/normalized counts.
    # No log1p here.
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)

    return adata


def harmonize_genes(adata_ref, adata_mov, n_top_genes=2000):
    """
    Keep shared genes, then choose HVGs jointly.
    """
    print("\nFinding shared genes...")
    common = adata_ref.var_names.intersection(adata_mov.var_names)
    if len(common) == 0:
        raise ValueError("No shared genes between reference and moving slices")

    adata_ref = adata_ref[:, common].copy()
    adata_mov = adata_mov[:, common].copy()

    print(f"Shared genes: {len(common)}")

    adata_all = sc.concat(
        {"ref": adata_ref, "mov": adata_mov},
        label="batch",
        index_unique="-"
    )

    n_top = min(n_top_genes, adata_all.shape[1])

    # seurat_v3 generally expects counts; after normalize_total it still often works,
    # but if your data are true counts, you could also skip normalize_total earlier.
    sc.pp.highly_variable_genes(
        adata_all,
        n_top_genes=n_top,
        flavor="seurat_v3",
        batch_key="batch"
    )

    hvg = adata_all.var_names[adata_all.var["highly_variable"]].tolist()
    if len(hvg) == 0:
        raise ValueError("No HVGs selected")

    print(f"HVG genes kept: {len(hvg)}")

    adata_ref = adata_ref[:, hvg].copy()
    adata_mov = adata_mov[:, hvg].copy()

    # enforce same order
    adata_mov = adata_mov[:, adata_ref.var_names].copy()

    same_order = np.array_equal(adata_ref.var_names, adata_mov.var_names)
    print(f"Gene order identical: {same_order}")
    if not same_order:
        raise ValueError("Gene order mismatch after harmonization")

    return adata_ref, adata_mov


def build_params(tag: str, use_gpu: bool):
    """
    Parameters closer to official CAST demo.
    """
    gpu_id = 0 if use_gpu else -1

    params = reg_params(
        dataname=tag,
        gpu=gpu_id,
        diff_step=5,

        # Affine
        iterations=500,
        dist_penalty1=0,
        bleeding=500,
        d_list=[3, 2, 1, 0.5, 1 / 3],
        attention_params=[None, 3, 1, 0],

        # FFD
        dist_penalty2=[0],
        alpha_basis_bs=[500],
        meshsize=[8],
        iterations_bs=[400],
        attention_params_bs=[[None, 3, 1, 0]],
        mesh_weight=[None],
    )

    # Same spirit as official demo
    params.alpha_basis = torch.tensor(
        [1 / 1000, 1 / 1000, 1 / 50, 5, 5],
        dtype=torch.float32,
        device=params.device
    ).reshape(5, 1)

    return params


def summarize_shift(coords_before: np.ndarray, coords_after: np.ndarray):
    disp = np.linalg.norm(coords_after - coords_before, axis=1)
    stats = {
        "n_cells": int(len(disp)),
        "mean_shift": float(np.mean(disp)),
        "median_shift": float(np.median(disp)),
        "p95_shift": float(np.quantile(disp, 0.95)),
        "max_shift": float(np.max(disp)),
    }
    return stats


def save_shift_stats(stats: dict, out_csv: str):
    pd.DataFrame([stats]).to_csv(out_csv, index=False)


def to_torch_dict(coords_ref, coords_mov, expr_ref, expr_mov):
    coords_raw = {
        "ref": coords_ref.astype(np.float32),
        "mov": coords_mov.astype(np.float32),
    }
    exp_dict = {
        "ref": torch.tensor(expr_ref, dtype=torch.float32),
        "mov": torch.tensor(expr_mov, dtype=torch.float32),
    }
    return coords_raw, exp_dict


def run_alignment(adata_ref, adata_mov, out_dir: str, tag: str):
    coords_ref = np.asarray(adata_ref.obsm["spatial"], dtype=np.float32)
    coords_mov = np.asarray(adata_mov.obsm["spatial"], dtype=np.float32)

    expr_ref = ensure_dense_float32(adata_ref.X)
    expr_mov = ensure_dense_float32(adata_mov.X)

    print(f"\nRunning CAST for {tag}")
    print(f"ref coords: {coords_ref.shape}, ref expr: {expr_ref.shape}")
    print(f"mov coords: {coords_mov.shape}, mov expr: {expr_mov.shape}")

    coords_raw_t, exp_dict_t = to_torch_dict(
        coords_ref, coords_mov, expr_ref, expr_mov
    )

    use_gpu = torch.cuda.is_available()
    gpu_t = 0 if use_gpu else -1
    print(f"GPU available: {use_gpu}")

    # CAST_MARK
    print("\n[1/2] Running CAST_MARK...")
    embed_dict = CAST_MARK(
        coords_raw_t=coords_raw_t,
        exp_dict_t=exp_dict_t,
        output_path_t=out_dir,
        task_name_t=tag,
        gpu_t=gpu_t,
        epoch_t=100,
        if_plot=False,
        graph_strategy="delaunay",
    )

    # CAST_STACK
    print("\n[2/2] Running CAST_STACK...")
    params_dist = build_params(tag, use_gpu)

    coords_final = CAST_STACK(
        coords_raw=coords_raw_t,
        embed_dict=embed_dict,
        output_path=out_dir,
        graph_list=["mov", "ref"],   # mov -> ref
        params_dist=params_dist,
        early_stop_thres=1e-5,
        renew_mesh_trans=False,
        rescale=True,
    )

    # Convert output to numpy
    out = {}
    for k, v in coords_final.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = np.asarray(v, dtype=np.float32)

    return out


def plot_alignment(coords_ref, coords_mov, coords_aligned, out_png):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(coords_ref[:, 0], coords_ref[:, 1], s=0.5, alpha=0.5, label="ref")
    plt.scatter(coords_mov[:, 0], coords_mov[:, 1], s=0.5, alpha=0.5, label="mov")
    plt.title("Before alignment")
    plt.legend(markerscale=6)

    plt.subplot(1, 2, 2)
    plt.scatter(coords_ref[:, 0], coords_ref[:, 1], s=0.5, alpha=0.5, label="ref")
    plt.scatter(coords_aligned[:, 0], coords_aligned[:, 1], s=0.5, alpha=0.5, label="aligned")
    plt.title("After alignment")
    plt.legend(markerscale=6)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    data_dir = get_env("CAST_DATA_DIR")
    out_dir = get_env("CAST_OUT_DIR")
    pairs_str = get_env("CAST_PAIRS")

    xcol = os.environ.get("CAST_XCOL", "center_x")
    ycol = os.environ.get("CAST_YCOL", "center_y")
    n_top_genes = int(os.environ.get("CAST_N_TOP_GENES", "2000"))

    os.makedirs(out_dir, exist_ok=True)
    pairs = parse_pairs(pairs_str)

    for meta_ref, meta_mov, expr_ref, expr_mov in pairs:
        tag = f"{meta_mov}_TO_{meta_ref}"
        print("\n" + "=" * 80)
        print(f"Running pair: {tag}")
        print("=" * 80)

        adata_ref = load_slice(
            data_dir=data_dir,
            meta_file=meta_ref,
            expr_file=expr_ref,
            xcol=xcol,
            ycol=ycol,
            name="ref",
        )

        adata_mov = load_slice(
            data_dir=data_dir,
            meta_file=meta_mov,
            expr_file=expr_mov,
            xcol=xcol,
            ycol=ycol,
            name="mov",
        )

        adata_ref, adata_mov = harmonize_genes(
            adata_ref,
            adata_mov,
            n_top_genes=n_top_genes,
        )

        coords_final = run_alignment(
            adata_ref=adata_ref,
            adata_mov=adata_mov,
            out_dir=out_dir,
            tag=tag,
        )

        mov_before = np.asarray(adata_mov.obsm["spatial"], dtype=np.float32)
        mov_after = np.asarray(coords_final["mov"], dtype=np.float32)
        ref_coords = np.asarray(adata_ref.obsm["spatial"], dtype=np.float32)

        out_csv = os.path.join(out_dir, f"{tag}_aligned.csv")
        out_qc = os.path.join(out_dir, f"{tag}_qc.png")
        out_stats = os.path.join(out_dir, f"{tag}_shift_stats.csv")

        pd.DataFrame({
            "cell_id": adata_mov.obs_names.astype(str),
            "x_aligned": mov_after[:, 0],
            "y_aligned": mov_after[:, 1],
        }).to_csv(out_csv, index=False)

        plot_alignment(
            coords_ref=ref_coords,
            coords_mov=mov_before,
            coords_aligned=mov_after,
            out_png=out_qc,
        )

        stats = summarize_shift(mov_before, mov_after)
        save_shift_stats(stats, out_stats)

        print("\nSaved outputs:")
        print(out_csv)
        print(out_qc)
        print(out_stats)
        print("Shift summary:", stats)


if __name__ == "__main__":
    main()