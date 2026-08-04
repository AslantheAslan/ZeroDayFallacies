import pandas as pd
from collections import defaultdict

dataset = 'NF-CSE-CIC-IDS2018-v3'

# Load CSV
df = pd.read_csv("summary_non_zero_day.csv")

if dataset == 'CIDDS':
    # Models and OOD classes in the order shown in the LaTeX table
    model_order = ["SVM", "MLP", "HeteroSage", "HeteroGAT", "HGT", "Proposed"]
    ood_order = ["bruteForce", "DoS", "pingScan", "portScan"]

else:
    model_order = ["SVM", "MLP", "HeteroSage", "HeteroGAT", "HGT", "Proposed"]
    ood_order = ["Bot", "GoldenEye", "Hulk", "Infiltration", "SlowHTTPTest", "Slowloris", "bruteForceWeb", "SQL",
                 "HOIC", "LOIC-UDP", "LOIC-HTTP", "bruteForceSSH", "bruteForceFTP", "XSS"]

# Mapping because CSV uses lowercase 'dos'
ood_normalize = {
    "dos": "DoS",
    "DoS": "DoS",
    "bruteForce": "bruteForce",
    "pingScan": "pingScan",
    "portScan": "portScan",
    "Bot": "Bot",
    "Brute_Force_-XSS": "XSS",
    "DoS_attacks-GoldenEye": "GoldenEye",
    "DoS_attacks-Hulk": "Hulk",
    "DoS_attacks-SlowHTTPTest": "SlowHTTPTest",
    "DoS_attacks-Slowloris": "Slowloris",
    "Brute_Force_-Web": "bruteForceWeb",
    "SQL_Injection": "SQL",
    "Infilteration": "Infiltration",
    "DDOS_attack-HOIC": "HOIC",
    "DDOS_attack-LOIC-UDP": "LOIC-UDP",
    "DDoS_attacks-LOIC-HTTP": "LOIC-HTTP",
    "SSH-Bruteforce": "bruteForceSSH",
    "FTP-BruteForce": "bruteForceFTP"
}

df["ood_class"] = df["ood_class"].map(ood_normalize)

# Metrics to extract
metrics = [
    ("FPR@95", "FPR@95"),
    ("MACRO-F1", "Macro-f1"),
    ("AUROC", "AUROC"),
    ("AUPR-In", "AUPR In"),
    ("AUPR-Out", "AUPR Out")
]

# Organize values by (model, ood_class, preprocess)
table = defaultdict(dict)

for _, row in df.iterrows():
    key = (row["model"], row["ood_class"])
    preprocess = row["preprocess"]  # balanced_raw or balanced_pretransform
    table[key][preprocess] = row

def get_metric_value(row, metric_col):
    """Return metric as LaTeX-friendly string, or '/' if missing."""
    if row is None:
        return "/"
    return row[metric_col] if metric_col in row else "/"

# ---------------------------------------------------------------------
# Generate LaTeX lines
# ---------------------------------------------------------------------

def format_row(model, ood):
    key = (model, ood)
    raw = table[key].get("balanced_noise_0.0_raw", None)
    noise = table[key].get("balanced_noise_0.05_pretransform", None)

    latex_values = []
    for metric_col, _ in metrics:
        left = get_metric_value(raw, metric_col)
        right = get_metric_value(noise, metric_col)
        latex_values.append(f"{left} / {right}")

    return latex_values


# ---------------------------------------------------------------------
# PRINT LATEX TABLE BODY
# ---------------------------------------------------------------------

for model in model_order:
    print("\\midrule")
    print(f"\\multirow{{4}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{model}}}}}}}")

    for ood in ood_order:
        row_values = format_row(model, ood)
        joined = " & ".join(row_values)
        print(f"& {ood} & {joined} \\\\")