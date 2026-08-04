import pandas as pd
from collections import defaultdict

dataset = 'NF-CSE-CIC-IDS2018-v3'
# Load CSV
df = pd.read_csv("summary_zero_day.csv")

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

# Metrics in CSV → metrics in LaTeX table
metrics = [
    ("FPR@95", "FPR@95"),
    ("FalseAlarmRate", "FAR"),
    ("ZeroDay-DetectionRate", "Detection Rate"),
    ("MACRO-F1", "Macro-f1"),
    ("AUROC", "AUROC"),
    ("AUPR-In", "AUPR In"),
    ("AUPR-Out", "AUPR Out")
]

# Map (model, ood_class, preprocess) → row
table = defaultdict(dict)
for _, row in df.iterrows():
    key = (row["model"], row["ood_class"])
    # print(f"{key}")
    # preprocess = row["preprocess"].split('_')[-1]  # balanced_raw / balanced_pretransform
    preprocess = row["preprocess"]
    table[key][preprocess] = row
    # print(table[key][preprocess])

def get_metric_value(row, col):
    """Return the metric value or '/' if missing."""
    if row is None:
        return "/"
    return row[col] if col in row else "/"

def format_row(model, ood):
    """Return list of LaTeX entries like 'x / y' per metric."""
    key = (model, ood)
    raw = table[key].get("balanced_noise_0.0_raw", None)
    noise = table[key].get("balanced_noise_0.05_pretransform", None)
    # print(noise)

    values = []
    for csv_col, _ in metrics:
        left = get_metric_value(raw, csv_col)
        right = get_metric_value(noise, csv_col)
        values.append(f"{left} / {right}")
    # print(values)
    return values

# ----------------------------
# PRINT LATEX ROWS
# ----------------------------

for model in model_order:
    print("\\midrule")
    print(f"\\multirow{{4}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{{model}}}}}}}")

    for ood in ood_order:
        row_values = format_row(model, ood)
        joined = " & ".join(row_values)
        print(f"& {ood} & {joined} \\\\")