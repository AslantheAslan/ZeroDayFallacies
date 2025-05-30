import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer


def preprocess_CIDDS() -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sketches insightful figures for CIDDS. Saves train, val, and test splits as parquet files. Adopted from
    https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python/blob/main/Chapter16/chapter16
    .ipynb
    :return: df_train, df_val, df_test
    """

    df = pd.read_csv("./data/CIDDS/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv")
    print("RAW DATAFRAME SHAPE: ", df.shape)
    print(df.head())

    # Drop unnecessary colums
    df = df.drop(columns=['Src Pt', 'Dst Pt', 'Flows', 'Tos', 'class', 'attackID', 'attackDescription'])
    df['attackType'] = df['attackType'].replace('---', 'benign')
    df['Date first seen'] = pd.to_datetime(df['Date first seen'])
    print("NEW DATAFRAME SHAPE: ", df.shape)
    print(df.head())

    # Count labels
    count_labels = df['attackType'].value_counts() / len(df) * 100
    print(count_labels)
    plt.pie(count_labels[:3], labels=df['attackType'].unique()[:3], autopct='%.0f%%')
    plt.savefig("count_label", dpi=300, bbox_inches='tight')

    # Feature Distribution
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 5))
    df['Duration'].hist(ax=ax1)
    ax1.set_xlabel("Duration")
    df['Packets'].hist(ax=ax2)
    ax2.set_xlabel("Number of packets")
    pd.to_numeric(df['Bytes'], errors='coerce').hist(ax=ax3)
    ax3.set_xlabel("Number of bytes")
    plt.savefig("feature_distribution", dpi=300, bbox_inches='tight')

    # Preprocessing the CIDDS-001 dataset

    df['weekday'] = df['Date first seen'].dt.weekday
    df = pd.get_dummies(df, columns=['weekday']).rename(columns={'weekday_0': 'Monday',
                                                                 'weekday_1': 'Tuesday',
                                                                 'weekday_2': 'Wednesday',
                                                                 'weekday_3': 'Thursday',
                                                                 'weekday_4': 'Friday',
                                                                 'weekday_5': 'Saturday',
                                                                 'weekday_6': 'Sunday',
                                                                 })

    df['daytime'] = (df['Date first seen'].dt.second + df['Date first seen'].dt.minute * 60 + df[
        'Date first seen'].dt.hour * 60 * 60) / (24 * 60 * 60)

    def one_hot_flags(input):
        return [1 if char1 == char2 else 0 for char1, char2 in zip('APRSF', input[1:])]

    df = df.reset_index(drop=True)
    ohe_flags = one_hot_flags(df['Flags'].to_numpy())
    ohe_flags = df['Flags'].apply(one_hot_flags).to_list()
    df[['ACK', 'PSH', 'RST', 'SYN', 'FIN']] = pd.DataFrame(ohe_flags, columns=['ACK', 'PSH', 'RST', 'SYN', 'FIN'])
    df = df.drop(columns=['Date first seen', 'Flags'])
    print("PREPROCESSED DATAFRAME SHAPE: ", df.shape)
    print(df.head())

    temp = pd.DataFrame()
    temp['SrcIP'] = df['Src IP Addr'].astype(str)
    temp['SrcIP'][~temp['SrcIP'].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
    temp = temp['SrcIP'].str.split('.', expand=True).rename(columns={2: 'ipsrc3', 3: 'ipsrc4'}).astype(int)[
        ['ipsrc3', 'ipsrc4']]
    temp['ipsrc'] = temp['ipsrc3'].apply(lambda x: format(x, "b").zfill(8)) + temp['ipsrc4'].apply(
        lambda x: format(x, "b").zfill(8))
    df = df.join(temp['ipsrc'].str.split('', expand=True)
                 .drop(columns=[0, 17])
                 .rename(columns=dict(enumerate([f'ipsrc_{i}' for i in range(17)])))
                 .astype('int32'))
    print(df.head(5))

    temp = pd.DataFrame()
    temp['DstIP'] = df['Dst IP Addr'].astype(str)
    temp['DstIP'][~temp['DstIP'].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
    temp = temp['DstIP'].str.split('.', expand=True).rename(columns={2: 'ipdst3', 3: 'ipdst4'}).astype(int)[
        ['ipdst3', 'ipdst4']]
    temp['ipdst'] = temp['ipdst3'].apply(lambda x: format(x, "b").zfill(8)) \
                    + temp['ipdst4'].apply(lambda x: format(x, "b").zfill(8))
    df = df.join(temp['ipdst'].str.split('', expand=True)
                 .drop(columns=[0, 17])
                 .rename(columns=dict(enumerate([f'ipdst_{i}' for i in range(17)])))
                 .astype('int32'))
    print(df.head(5))

    m_index = df[pd.to_numeric(df['Bytes'], errors='coerce').isnull() == True].index
    df['Bytes'].loc[m_index] = df['Bytes'].loc[m_index].apply(lambda x: 10e6 * float(x.strip().split()[0]))
    df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce', downcast='integer')

    df = pd.get_dummies(df, prefix='', prefix_sep='', columns=['Proto', 'attackType'])
    print(df.head(5))

    labels = ['benign', 'bruteForce', 'dos', 'pingScan', 'portScan']
    df_train, df_test = train_test_split(df, random_state=0, test_size=0.2, stratify=df[labels])
    df_val, df_test = train_test_split(df_test, random_state=0, test_size=0.5, stratify=df_test[labels])

    scaler = PowerTransformer()
    df_train[['Duration', 'Packets', 'Bytes']] = scaler.fit_transform(df_train[['Duration', 'Packets', 'Bytes']])
    df_val[['Duration', 'Packets', 'Bytes']] = scaler.transform(df_val[['Duration', 'Packets', 'Bytes']])
    df_test[['Duration', 'Packets', 'Bytes']] = scaler.transform(df_test[['Duration', 'Packets', 'Bytes']])

    df_train.to_parquet('./data/CIDDS/df_train.parquet')
    df_val.to_parquet('./data/CIDDS/df_val.parquet')
    df_test.to_parquet('./data/CIDDS/df_test.parquet')

    print(df_train[df_train['benign'] == 1].head())

    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 5))
    df_train['Duration'].hist(ax=ax1)
    ax1.set_xlabel("Duration")
    df_train['Packets'].hist(ax=ax2)
    ax2.set_xlabel("Number of packets")
    df_train['Bytes'].hist(ax=ax3)
    ax3.set_xlabel("Number of bytes")
    plt.savefig("scaled_feature_distribution", dpi=300, bbox_inches='tight')

    return df_train, df_val, df_test
