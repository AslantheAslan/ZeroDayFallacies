import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score
import glob
import os
from joblib import Parallel, delayed   # <--- ADDED

# -----------------------------
# Metric computation functions
# -----------------------------
def evaluate_continuous_scores(y_true, scores, benign_class=0, invert_score=False):
    y_true_bin = np.where(y_true == benign_class, 0, 1)
    scores_proc = -scores if invert_score else scores
    if len(np.unique(y_true_bin)) < 2:
        return {"AUROC": np.nan, "FPR@95": np.nan, "AUPR-In": np.nan, "AUPR-Out": np.nan}

    fpr, tpr, _ = roc_curve(y_true_bin, scores_proc)
    auroc = roc_auc_score(y_true_bin, scores_proc)
    fpr_at_95 = fpr[np.argmin(np.abs(tpr - 0.95))]

    prec_out, rec_out, _ = precision_recall_curve(y_true_bin, scores_proc)
    aupr_out = auc(rec_out, prec_out)

    prec_in, rec_in, _ = precision_recall_curve(1 - y_true_bin, -scores_proc)
    aupr_in = auc(rec_in, prec_in)

    return {"AUROC": auroc, "FPR@95": fpr_at_95, "AUPR-In": aupr_in, "AUPR-Out": aupr_out}


def evaluate_classification(y_true, y_pred, benign_class=0):
    y_true_bin = np.where(y_true == benign_class, 0, 1)
    y_pred_bin = np.where(y_pred == benign_class, 0, 1)

    if len(np.unique(y_true_bin)) < 2:
        return {"MACRO-F1": np.nan}

    f1 = f1_score(y_true_bin, y_pred_bin, average='macro')
    weighted_f1 = f1_score(y_true_bin, y_pred_bin, average='weighted')
    f1_per_class = f1_score(y_true_bin, y_pred_bin, average=None)
    f1_dict = {cls: score for cls, score in zip([0, 1], f1_per_class)}

    return {"MACRO-F1": f1,
            "Weighted-F1": weighted_f1,
            "benign-F1": f1_dict[0],
            "attack-F1": f1_dict[1]}


# -----------------------------
# Evaluate all runs per file
# -----------------------------
def evaluate_all_runs_dual(df, benign_class=0, zero_day_class=4, num_runs=10):
    model_results_nonzero = []
    model_results_zero = []

    pred_cols = [c for c in df.columns if c.startswith("predicted_label")]
    if 'score_1' in df.columns:
        maha_cols = [c for c in df.columns if c.startswith("score")]
    else:
        maha_cols = [c for c in df.columns if c.startswith("mahalanobis_score")]

    num_runs = min(len(pred_cols), len(maha_cols))

    if num_runs == 0:
        return {"non_zero_day": {}, "zero_day": {}}

    for i in range(num_runs):
        pred_col = pred_cols[i]
        maha_col = maha_cols[i]

        df_nonzero = df[df["true_label"] != zero_day_class]
        df_zero_eval = df[df["true_label"].isin([benign_class, zero_day_class])]

        if df_nonzero.empty or df_zero_eval.empty:
            continue

        # NON-ZERODAY
        y_true_nz = df_nonzero["true_label"].values
        y_pred_nz = df_nonzero[pred_col].values
        scores_nz = df_nonzero[maha_col].values

        metrics_model_nz = evaluate_classification(y_true_nz, y_pred_nz, benign_class)
        metrics_maha_nz = evaluate_continuous_scores(y_true_nz, scores_nz, benign_class)
        model_results_nonzero.append(metrics_model_nz | metrics_maha_nz)

        # ZERO-DAY vs BENIGN
        y_true_z = df_zero_eval["true_label"].values
        y_pred_z = df_zero_eval[pred_col].values
        scores_z = df_zero_eval[maha_col].values

        zero_day_mask = y_true_z == zero_day_class
        num_zero_day = np.sum(zero_day_mask)

        benign_mask = y_true_z == benign_class
        num_benign = np.sum(benign_mask)

        benign_false_alarm_rate = np.sum(y_pred_z[benign_mask] != benign_class) / max(num_benign, 1)
        zero_day_detection_rate = np.sum(y_pred_z[zero_day_mask] != benign_class) / max(num_zero_day, 1) if num_zero_day > 0 else np.nan

        metrics_model_z = evaluate_classification(y_true_z, y_pred_z, benign_class)
        metrics_model_z["ZeroDay-DetectionRate"] = zero_day_detection_rate
        metrics_model_z["FalseAlarmRate"] = benign_false_alarm_rate
        metrics_maha_z = evaluate_continuous_scores(y_true_z, scores_z, benign_class)

        model_results_zero.append(metrics_model_z | metrics_maha_z)

    df_nonzero = pd.DataFrame(model_results_nonzero)
    df_zero = pd.DataFrame(model_results_zero)

    summary = {
        "non_zero_day": df_nonzero.mean().to_dict() | {f"{k}_std": v for k, v in df_nonzero.std().to_dict().items()},
        "zero_day": df_zero.mean().to_dict() | {f"{k}_std": v for k, v in df_zero.std().to_dict().items()},
    }

    return summary


# -----------------------------
# Parse filename
# -----------------------------
def parse_filename(filename):
    base = os.path.basename(filename)
    print(base)
    model = base.split("benchmark_NF-CSE-CIC-IDS2018-v3_")[1].split("_epochs")[0]
    ood_class = base.split("['")[1].split("']")[0]
    preprocess = base.split("]")[1].split("__")[0]
    return model, ood_class, preprocess


# -------------------------------------------
# 🌟 NEW: Per-file processing for parallel use
# -------------------------------------------
def process_single_file(f, benign_class, zero_day_class):
    model, ood_class, preprocess = parse_filename(f)
    df = pd.read_parquet(f)
    summary = evaluate_all_runs_dual(df, benign_class, zero_day_class)

    if not summary["non_zero_day"] and not summary["zero_day"]:
        return None

    nz = summary["non_zero_day"]
    nz.update({"model": model, "ood_class": ood_class, "preprocess": preprocess})

    z = summary["zero_day"]
    z.update({"model": model, "ood_class": ood_class, "preprocess": preprocess})

    return nz, z


# -----------------------------
# Main pipeline (parallelized)
# -----------------------------
def main(parquet_dir="results/dataframes", benign_class=0, zero_day_class=14):
    files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    print(f"Found {len(files)} parquet files.")

    # ---------- 🔥 RUN IN PARALLEL ----------
    results = Parallel(n_jobs=8, backend="loky")(
        delayed(process_single_file)(f, benign_class, zero_day_class)
        for f in files
    )
    # ----------------------------------------

    all_results_nonzero = []
    all_results_zero = []

    for r in results:
        if r is None:
            continue
        nz, z = r
        all_results_nonzero.append(nz)
        all_results_zero.append(z)

    df_nonzero = pd.DataFrame(all_results_nonzero)
    df_zero = pd.DataFrame(all_results_zero)

    def merge_mean_std_latex(df):
        df_out = df.copy()
        for col in list(df.columns):
            if col.endswith("_std"):
                base = col[:-4]
                if base in df.columns:
                    df_out[base] = df_out.apply(
                        lambda r: (
                            f"{100*r[base]:.1f} $\\pm$ {100*r[col]:.1f}"
                            if pd.notna(r[base])
                            else ""
                        ),
                        axis=1
                    )
                    df_out.drop(columns=[col], inplace=True)
        return df_out

    df_nonzero_fmt = merge_mean_std_latex(df_nonzero)
    df_zero_fmt = merge_mean_std_latex(df_zero)

    df_nonzero_fmt.to_csv("summary_non_zero_day.csv", index=False)
    df_zero_fmt.to_csv("summary_zero_day.csv", index=False)

    print("\n✅ Done. Parallel summaries saved.")


if __name__ == "__main__":
    main(parquet_dir="results/dataframes")
