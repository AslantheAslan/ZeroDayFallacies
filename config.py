import argparse
from utils.utils import parse_class_weight

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIDDS', choices=['CIDDS', 'NF-UNSW-NB15-v3', 'NF-CSE-CIC-IDS2018-v3'])
    parser.add_argument('--perturbed', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model', type=str, default='HeteroSage', choices=['HeteroSage', 'MLP', 'LSTM', 'Transformer',
                                                                            'HeteroSageMulti', 'HeteroGAT', 'HGT', 'SVM'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--unknown_classes', nargs='+', type=str,
                        choices=['bruteForce', 'dos', 'pingScan', 'portScan','Analysis', 'Backdoor', 'Benign', 'DoS',
                                 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms',
                                 'Infilteration', 'Bot', 'ddos', 'DoS_attacks-GoldenEye', 'DoS_attacks-Hulk',
                                 'DoS_attacks-SlowHTTPTest', 'DoS_attacks-Slowloris', 'SQL_Injection',
                                 'Brute_Force_-Web', 'Brute_Force_-XSS', 'DDOS_attack-HOIC', 'DDOS_attack-LOIC-UDP',
                                 "DDoS_attacks-LOIC-HTTP", "FTP-BruteForce", "SSH-Bruteforce"],
                        help='Classes excluded during training')
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument(
        "--class_weight", type=parse_class_weight, default=None,
        help="Class weights as comma-separated floats (e.g., 1.0,0.5,2.0), or 'balanced', or None"
    )
    parser.add_argument('--pre_transform', action='store_true', default=False,
                        help='Apply data transformation before training')

    parser.add_argument('--mahalanobis', action='store_true', default=False,
                        help='Enable Mahalanobis distance computation for open-set recognition')
    parser.add_argument('--master_model', action='store_true', default=False,
                        help='Train master model with worker model embeddings')
    parser.add_argument('--path', type=str, default='', help='Model path for the pretrained model (with data '
                                                             'transformations)')
    parser.add_argument('--path_raw', type=str, default='', help='Model path for the pretrained model (no data '
                                                                 'transformations)')
    parser.add_argument('--master_arch', type=str, default='MLP', choices=['MLP', 'GatedMLP', 'ExtendedGatedMLP'])
    parser.add_argument('--noise', type=float, default=0.0)
    return parser.parse_args()
