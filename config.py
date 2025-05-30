import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIDDS', choices=['CIDDS'])
    parser.add_argument('--perturbed', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model', type=str, default='HeteroSage', choices=['HeteroSage', 'MLP', 'LSTM', 'Transformer'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--unknown_classes', nargs='+', type=str,
                        choices=['bruteForce', 'dos', 'pingScan', 'portScan'],
                        help='Classes excluded during training')
    parser.add_argument('--num_runs', type=int, default=1)
    return parser.parse_args()
