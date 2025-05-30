import pandas as pd
from data.download import download_CIDDS
from data.preprocess import preprocess_CIDDS

import os


def get_dataset(args):
    df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if args.dataset == 'CIDDS':
        labels = ['benign', 'bruteForce', 'dos', 'pingScan', 'portScan']
        download_CIDDS()
        if os.path.exists('./data/CIDDS/df_train.parquet'):
            print('✅  CIDDS dataset was already downloaded')
            df_train = pd.read_parquet('./data/CIDDS/df_train.parquet')
            df_val = pd.read_parquet('./data/CIDDS/df_val.parquet')
            df_test = pd.read_parquet('./data/CIDDS/df_test.parquet')
            print("✅  Fetched the splits that already exist")
        else:
            print("📦  Preprocessing data...")
            df_train, df_val, df_test = preprocess_CIDDS()
            print("✅  Preprocessed CIDDS data set and saved in parquet format")

        features_host = [f'ipsrc_{i}' for i in range(1, 17)] + [f'ipdst_{i}' for i in range(1, 17)]
        features_flow = ['daytime', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Duration', 'Packets',
                         'Bytes',
                         'ACK', 'PSH', 'RST', 'SYN', 'FIN', 'ICMP ', 'IGMP ', 'TCP  ', 'UDP  ']
    else:
        # Placeholder for future datasets
        labels = []
        features_host, features_flow = [], []
        df_train, df_val, df_test = [], [], []
        BATCH_SIZE = args.batch_size
        print("Dataset doesn't exists!!!")

    return df_train, df_val, df_test, features_host, features_flow, labels
