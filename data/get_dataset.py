import pandas as pd
from data.download import download_CIDDS
from data.preprocess import preprocess_CIDDS, preprocess_nf_unsw_nb15_v3_compact, wo_port_preprocess_nf_unsw_nb15_v3_compact

import os
import json


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

    elif args.dataset == 'NF-UNSW-NB15-v3':
        labels_path = "./data/NF-UNSW-NB15-v3-compact/attack_labels.json"
        with open(labels_path, "r") as f:
            labels = json.load(f)

        if os.path.exists('./data/NF-UNSW-NB15-v3-compact/df_train.parquet'):
            print('✅  NF-UNSW-NB15-v3 dataset was already downloaded')
            df_train = pd.read_parquet('./data/NF-UNSW-NB15-v3-compact/df_train.parquet')
            df_val = pd.read_parquet('./data/NF-UNSW-NB15-v3-compact/df_val.parquet')
            df_test = pd.read_parquet('./data/NF-UNSW-NB15-v3-compact/df_test.parquet')
            print("✅  Fetched the splits that already exist")
        else:
            print("📦  Preprocessing data...")
            wo_port_preprocess_nf_unsw_nb15_v3_compact(
                raw_path="./data/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.parquet",
                out_dir="./data/NF-UNSW-NB15-v3-compact",
                random_state=0,
            )
            df_train = pd.read_parquet('./data/NF-UNSW-NB15-v3-compact/df_train.parquet')
            df_val = pd.read_parquet('./data/NF-UNSW-NB15-v3-compact/df_val.parquet')
            df_test = pd.read_parquet('./data/NF-UNSW-NB15-v3-compact/df_test.parquet')
            print("✅  Preprocessed NF-UNSW-NB15-v3 data set and saved in parquet format")

        features_host = [
            "srcip_o1", "srcip_o2", "srcip_o3", "srcip_o4",
            "dstip_o1", "dstip_o2", "dstip_o3", "dstip_o4"
        ]

        # Weekday columns: only those present (you intentionally dropped absent days)
        # weekday_cols = [c for c in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        #                 if c in df_train.columns]

        # Flow features (compact, no explosion)
        # features_flow = [
        #     "daytime",
        #     *weekday_cols,
        #     "Duration", "Packets", "Bytes",
        #     "PROTOCOL", "L4_SRC_PORT", "L4_DST_PORT", "L7_PROTO",
        #     "IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES",
        #     "BYTE_IN_RATIO", "PKT_IN_RATIO",
        #     "IS_TCP", "TCP_FLAGS_RAW", "TCP_FLAG_COUNT",
        #     "FIN", "SYN", "RST", "PSH", "ACK",
        #     "MIN_TTL", "MAX_TTL",
        #     "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
        #     "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
        # ]

        # Keep only columns that actually exist (defensive)
        features_host = [c for c in features_host if c in df_train.columns]
        # features_flow = [c for c in features_flow if c in df_train.columns]
        excluded = set(labels) | {"Src IP Addr", "Dst IP Addr"} | set(features_host)
        features_flow = [c for c in df_train.columns if c not in excluded]

    elif args.dataset == 'NF-CSE-CIC-IDS2018-v3':
        labels_path = "./data/NF-CSE-CIC-IDS2018-v3/attack_labels.json"
        with open(labels_path, "r") as f:
            labels = json.load(f)

        if os.path.exists('./data/NF-CSE-CIC-IDS2018-v3/df_train.parquet'):
            print('✅  NF-CSE-CIC-IDS2018-v3 dataset was already downloaded')
            df_train = pd.read_parquet('./data/NF-CSE-CIC-IDS2018-v3/df_train.parquet')
            df_val = pd.read_parquet('./data/NF-CSE-CIC-IDS2018-v3/df_val.parquet')
            df_test = pd.read_parquet('./data/NF-CSE-CIC-IDS2018-v3/df_test.parquet')

            print("✅  Fetched the splits that already exist")
        else:
            print("📦  Preprocessing data...")
            wo_port_preprocess_nf_unsw_nb15_v3_compact(
                raw_path="./data/NF-CSE-CIC-IDS2018-v3/NF-CICIDS2018-v3.parquet",
                out_dir="./data/NF-CSE-CIC-IDS2018-v3",
                random_state=0,
            )
            df_train = pd.read_parquet('./data/NF-CSE-CIC-IDS2018-v3/df_train.parquet')
            df_val = pd.read_parquet('./data/NF-CSE-CIC-IDS2018-v3/df_val.parquet')
            df_test = pd.read_parquet('./data/NF-CSE-CIC-IDS2018-v3/df_test.parquet')

            print("✅  Preprocessed NF-CSE-CIC-IDS2018-v3 data set and saved in parquet format")

        features_host = [
            "srcip_o1", "srcip_o2", "srcip_o3", "srcip_o4",
            "dstip_o1", "dstip_o2", "dstip_o3", "dstip_o4"
        ]

        # Keep only columns that actually exist
        features_host = [c for c in features_host if c in df_train.columns]
        # features_flow = [c for c in features_flow if c in df_train.columns]
        excluded = set(labels) | {"Src IP Addr", "Dst IP Addr", "L4_SRC_PORT", "L4_DST_PORT", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"} | set(features_host)
        features_flow = [c for c in df_train.columns if c not in excluded]

    else:
        # Placeholder for future datasets
        labels = []
        features_host, features_flow = [], []
        df_train, df_val, df_test = [], [], []
        BATCH_SIZE = args.batch_size
        print("Dataset doesn't exists!!!")

    return df_train, df_val, df_test, features_host, features_flow, labels
