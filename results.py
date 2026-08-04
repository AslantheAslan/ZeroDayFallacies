import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, auc,
    f1_score
)
import warnings


def evaluate_continuous_scores(y_true, scores, benign_class=0, invert_score=False):
    """Compute AUROC, FPR@95, AUPR-In, AUPR-Out for continuous Mahalanobis scores."""
    y_true_bin = np.where(y_true == benign_class, 0, 1)
    scores_proc = -scores if invert_score else scores

    # Skip invalid cases (only one class present)
    if len(np.unique(y_true_bin)) < 2:
        return {"AUROC": np.nan, "FPR@95": np.nan, "AUPR-In": np.nan, "AUPR-Out": np.nan}

    # ROC metrics
    fpr, tpr, _ = roc_curve(y_true_bin, scores_proc)
    auroc = roc_auc_score(y_true_bin, scores_proc)
    fpr_at_95 = fpr[np.argmin(np.abs(tpr - 0.95))]

    # AUPR-Out (malicious = positive)
    prec_out, rec_out, _ = precision_recall_curve(y_true_bin, scores_proc)
    aupr_out = auc(rec_out, prec_out)

    # AUPR-In (benign = positive)
    prec_in, rec_in, _ = precision_recall_curve(1 - y_true_bin, -scores_proc)
    aupr_in = auc(rec_in, prec_in)

    return {
        "AUROC": auroc,
        "FPR@95": fpr_at_95,
        "AUPR-In": aupr_in,
        "AUPR-Out": aupr_out
    }


def evaluate_classification(y_true, y_pred, benign_class=0):
    """Compute F1-score treating benign as 0 and malicious as 1."""
    y_true_bin = np.where(y_true == benign_class, 0, 1)
    y_pred_bin = np.where(y_pred == benign_class, 0, 1)

    if len(np.unique(y_true_bin)) < 2:
        return {"MACRO-F1": np.nan}

    f1 = f1_score(y_true_bin, y_pred_bin, average='macro')
    weighted_f1 = f1_score(y_true_bin, y_pred_bin, average='weighted')

    # Compute per-class F1
    f1_per_class = f1_score(y_true_bin, y_pred_bin, average=None)
    print(f1_per_class)
    f1_dict = {cls: score for cls, score in zip([0, 1], f1_per_class)}

    return {"MACRO-F1": f1,
            "Weighted-F1": weighted_f1,
            "benign-F1": f1_dict[0],
            "attack-F1": f1_dict[1]
            }


def evaluate_all_runs_dual(df, benign_class=0, zero_day_class=4, num_runs=10):
    """Evaluate both normal and Mahalanobis-based predictions for multiple runs."""

    model_results_nonzero = []
    model_results_zero = []

    for i in range(1, num_runs + 1):
        # Dynamic column names
        pred_col = f"predicted_label_{i}"
        maha_col = f"mahalanobis_score_{i}"
        score_col = f"score_{i}"
        thr_col = f"threshold_{i}"
        pred_maha_col = f"y_pred_maha_{i}"

        if 'score_1' in df.columns:
            maha_col = score_col

        # Subsets
        df_nonzero = df[df["true_label"] != zero_day_class]
        df_zero_eval = df[df["true_label"].isin([benign_class, zero_day_class])]

        # Skip empty sets
        if df_nonzero.empty or df_zero_eval.empty:
            continue

        # ========== NON-ZERO-DAY (BENIGN + CLASSES 1–3) ==========
        y_true_nz = df_nonzero["true_label"].values
        y_pred_nz = df_nonzero[pred_col].values
        scores_nz = df_nonzero[maha_col].values

        metrics_model_nz = evaluate_classification(y_true_nz, y_pred_nz, benign_class)
        metrics_maha_nz = evaluate_continuous_scores(y_true_nz, scores_nz, benign_class)

        model_results_nonzero.append(metrics_model_nz | metrics_maha_nz)

        # ========== ZERO-DAY vs BENIGN ==========
        y_true_z = df_zero_eval["true_label"].values
        y_pred_z = df_zero_eval[pred_col].values
        scores_z = df_zero_eval[maha_col].values

        zero_day_mask = y_true_z == zero_day_class
        num_zero_day = np.sum(zero_day_mask)

        benign_mask = y_true_z == benign_class
        num_benign = np.sum(benign_mask)

        benign_pred_non_benign = np.sum(y_pred_z[benign_mask] != benign_class)
        benign_false_alarm_rate = benign_pred_non_benign / num_benign

        if num_zero_day > 0:
            detected_as_non_benign = np.sum(y_pred_z[zero_day_mask] != benign_class)
            zero_day_detection_rate = detected_as_non_benign / num_zero_day
        else:
            zero_day_detection_rate = np.nan

        metrics_model_z = evaluate_classification(y_true_z, y_pred_z, benign_class)
        metrics_model_z["ZeroDay-DetectionRate (ZDDR)"] = zero_day_detection_rate
        metrics_model_z["FalseAlarmRate (FAR)"] = benign_false_alarm_rate

        metrics_maha_z = evaluate_continuous_scores(y_true_z, scores_z, benign_class)

        model_results_zero.append(metrics_model_z | metrics_maha_z)

    # Convert to DataFrames for easy mean/std
    df_nonzero = pd.DataFrame(model_results_nonzero)
    df_zero = pd.DataFrame(model_results_zero)

    summary = {
        "non_zero_day": df_nonzero.mean().to_dict() | {f"{k}_std": v for k, v in df_nonzero.std().to_dict().items()},
        "zero_day": df_zero.mean().to_dict() | {f"{k}_std": v for k, v in df_zero.std().to_dict().items()},
    }

    return summary


def evaluate_single_run(
    parquet_path,
    run=1,
    benign_class=0,
    zero_day_class=4,
    dataset='CIDDS',
    score_column=None,
    is_kumo=False,
    zero_day_label=None,
):
    """Evaluate both normal and Mahalanobis-based predictions for a single run."""
    df = pd.read_parquet(parquet_path).copy()

    if not is_kumo:
        pred_col = f"predicted_label_{run}"
        if score_column is None:
            score_column = f"score_{run}" if f"score_{run}" in df.columns else f"mahalanobis_score_{run}"
        true_col = "true_label"

    else:
        if zero_day_label is None:
            raise ValueError("When is_kumo=True, you must provide zero_day_label")

        # --- Step 1: define all labels ---
        if dataset == 'CIDDS':
            all_labels = ["benign", "dos", "portScan", "pingScan", "bruteForce"]
        elif dataset == 'NF-CSE-CIC-IDS2018-v3':
            all_labels = [
                "benign",
                "Bot",
                "Brute_Force_-Web",
                "Brute_Force_-XSS",
                "DDOS_attack-HOIC",
                "DDOS_attack-LOIC-UDP",
                "DDoS_attacks-LOIC-HTTP",
                "DoS_attacks-GoldenEye",
                "DoS_attacks-Hulk",
                "DoS_attacks-SlowHTTPTest",
                "DoS_attacks-Slowloris",
                "FTP-BruteForce",
                "Infilteration",
                "SQL_Injection",
                "SSH-Bruteforce",
            ]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if zero_day_label not in all_labels:
            raise ValueError(f"Unknown zero_day_label: {zero_day_label}")

        # --- Step 2: build mapping with unique class IDs ---
        label_map = {lbl: i for i, lbl in enumerate(all_labels)}

        # IMPORTANT:
        # When is_kumo=True, zero_day_class should come from zero_day_label.
        # This avoids forcing the zero-day class to 4 and colliding with another class.
        zero_day_class = label_map[zero_day_label]

        # --- Step 3: apply mapping ---
        df["y_true"] = df["y_true"].map(label_map)
        df["y_pred"] = df["y_pred"].map(label_map)

        # Safety checks
        if df["y_true"].isna().any():
            raise ValueError("Unknown labels found in y_true")

        if df["y_pred"].isna().any():
            raise ValueError("Unknown labels found in y_pred")

        df["y_true"] = df["y_true"].astype(int)
        df["y_pred"] = df["y_pred"].astype(int)

        pred_col = "y_pred"
        true_col = "y_true"

        if score_column is None:
            score_column = "score_benign"

    score_col = score_column

    if is_kumo:
        if score_col not in df.columns:
            avail = [c for c in df.columns if c.startswith("score_")]
            raise ValueError(f"Score column {score_col!r} missing. Available: {avail}")
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)

    # Subsets
    df_nonzero = df[df[true_col] != zero_day_class]
    df_zero_eval = df[df[true_col].isin([benign_class, zero_day_class])]

    if df_nonzero.empty or df_zero_eval.empty:
        raise ValueError("One of the evaluation subsets is empty.")

    # ========== NON-ZERO-DAY ==========
    y_true_nz = df_nonzero[true_col].values
    y_pred_nz = df_nonzero[pred_col].values

    if not is_kumo:
        scores_nz = df_nonzero[score_col].values
    else:
        scores_nz = 1 - df_nonzero[score_col].values  # benign → attack

    metrics_model_nz = evaluate_classification(y_true_nz, y_pred_nz, benign_class)
    metrics_maha_nz = evaluate_continuous_scores(y_true_nz, scores_nz, benign_class)

    non_zero_day_results = metrics_model_nz | metrics_maha_nz

    # ========== ZERO-DAY ==========
    y_true_z = df_zero_eval[true_col].values
    y_pred_z = df_zero_eval[pred_col].values

    if not is_kumo:
        scores_z = df_zero_eval[score_col].values
    else:
        scores_z = 1 - df_zero_eval[score_col].values

    zero_day_mask = y_true_z == zero_day_class
    benign_mask = y_true_z == benign_class

    num_zero_day = np.sum(zero_day_mask)
    num_benign = np.sum(benign_mask)

    benign_false_alarm_rate = (
        np.sum(y_pred_z[benign_mask] != benign_class) / num_benign
        if num_benign > 0 else np.nan
    )

    zero_day_detection_rate = (
        np.sum(y_pred_z[zero_day_mask] != benign_class) / num_zero_day
        if num_zero_day > 0 else np.nan
    )

    metrics_model_z = evaluate_classification(y_true_z, y_pred_z, benign_class)
    metrics_model_z["ZeroDay-DetectionRate (ZDDR)"] = zero_day_detection_rate
    metrics_model_z["FalseAlarmRate (FAR)"] = benign_false_alarm_rate

    metrics_maha_z = evaluate_continuous_scores(y_true_z, scores_z, benign_class)

    zero_day_results = metrics_model_z | metrics_maha_z

    return {
        "non_zero_day": non_zero_day_results,
        "zero_day": zero_day_results,
    }


def evaluate_parquet_results(parquet_path, benign_class=0, zero_day_class=4, num_runs=10):
    """Main function: loads a parquet file and computes all metrics."""
    print(f"📂 Loading results from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print("🔹 Evaluating known (non-zero-day) classes...")
    print("🔸 Evaluating zero-day performance...")

    results = evaluate_all_runs_dual(df, benign_class, zero_day_class, num_runs)

    print("\n=== 📊 Summary ===")
    for subset, metrics in results.items():
        print(f"\n▶ {subset.upper()}")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}" if not np.isnan(v) else f"  {k:12s}: NaN")

    return results


# Example usage
if __name__ == "__main__":

    # artifacts/run_1/kumo_cidds_loo/query_test__loo_bruteForce/predictions_merged.parquet
    # artifacts/run_1/kumo_cidds_loo_noise/query_test__loo_bruteForce__noise_m0_s0.05_seed42__flows/predictions_merged.parquet
    # artifacts/run_1/kumo_cidds_loo/query_test__loo_dos/predictions_merged.parquet
    # artifacts/run_1/kumo_cidds_loo_noise/query_test__loo_dos__noise_m0_s0.05_seed42__flows/predictions_merged.parquet
    # artifacts/run_1/kumo_cidds_loo/query_test__loo_pingScan/predictions_merged.parquet
    # artifacts/run_1/kumo_cidds_loo_noise/query_test__loo_pingScan__noise_m0_s0.05_seed42__flows/predictions_merged.parquet
    # artifacts/run_1/kumo_cidds_loo/query_test__loo_portScan/predictions_merged.parquet
    # artifacts/run_1/kumo_cidds_loo_noise/query_test__loo_portScan__noise_m0_s0.05_seed42__flows/predictions_merged.parquet




    # parquet_file = 'results/dataframes/benchmark_SVM_epochs_10_lr_0.001_out_dim_4_hidden_dim_64_num_layers_2[\'bruteForce\']balanced_raw_.parquet'
    # results = evaluate_parquet_results(parquet_file)
    # parquet_file = 'results/dataframes/benchmark_CIDDS_HeteroSage_epochs_100_lr_0.001_out_dim_4_hidden_dim_64_num_layers_10[\'dos\']balanced_noise_0.05_pretransform__mahalanobis_.parquet'
    # parquet_file = 'artifacts/run_3/kumo_cidds_loo_noise/query_test__loo_portScan__noise_m0_s0.05_seed42__flows/predictions_merged.parquet'
    # results = evaluate_single_run(parquet_file, run=1, is_kumo=True, zero_day_label='portScan')
    # parquet_path = 'results/dataframes/full_set_benchmark_NF-CSE-CIC-IDS2018-v3_HeteroSage_epochs_100_lr_0.001_out_dim_14_hidden_dim_64_num_layers_10[\'Infilteration\']balanced_noise_0.0_raw__mahalanobis_.parquet'
    # results = evaluate_parquet_results(parquet_path, zero_day_class=14) # zero_day_class = 4 for CIDDS, 14 for CICIDS2018

    #parquet_file = 'artifacts/kumo_rfm_loo/NF-CSE-CIC-IDS2018-v3/query_test__loo_Infilteration/predictions_merged.parquet'
    parquet_file = 'artifacts/kumo_rfm_loo/NF-CSE-CIC-IDS2018-v3/query_test__loo_Brute_Force_-XSS/predictions_merged.parquet'
    results = evaluate_single_run(parquet_file, dataset='NF-CSE-CIC-IDS2018-v3', run=1, is_kumo=True, zero_day_label='Brute_Force_-XSS')
    for name, df_sum in results.items():
        print(f"\n=== {name} ===")
        print(df_sum)
