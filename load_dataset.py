import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.model_selection import train_test_split

urls = [
        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names"
        ]


def load_kdd_to_df():
    # load th kdd to data frame
    # return normal sample data and label ans attack data and label
    df_colnames = pd.read_csv(urls[1], skiprows=1, sep=':', names=['names', 'types'])
    df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']
    df = pd.read_csv(urls[0], header=None, names=df_colnames['names'].values)
    df_symbolic = df_colnames[df_colnames['types'].str.contains('symbolic.')]
    df_continuous = df_colnames[df_colnames['types'].str.contains('continuous.')]
    samples = pd.get_dummies(df.iloc[:, :-1], columns=df_symbolic['names'][:-1])
    labels = np.where(df['status'] == 'normal.', 1, 0)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(samples)
    norm_samples = df_scaled[labels == 1]  # normal data
    attack_samples = df_scaled[labels == 0]  # attack data

    norm_labels = labels[labels == 1]
    attack_labels = labels[labels == 0]

    return norm_samples, norm_labels, attack_samples, attack_labels


def load_dataset(train_precent=0.8, seed=None):

    # split data to train and test , train data consist only attack data
    random.seed(seed)
    norm_samples, norm_labels, attack_samples, attack_labels = load_kdd_to_df()

    X_train, x_attack_test = train_test_split(attack_samples, random_state=seed, test_size=1-train_precent)

    # generate test set consist of 50% attack and 50% normal

    len_attack_test = x_attack_test.shape[0]
    if norm_samples.shape[0] > len_attack_test :
        norm_test_samples = pd.DataFrame(norm_samples).sample(n=len_attack_test, random_state=seed).to_numpy()
    else:
        norm_test_samples = pd.DataFrame(norm_samples).sample(n=norm_samples.shape[0], random_state=seed).to_numpy()

    X_test = np.concatenate([x_attack_test, norm_test_samples])

    y_test = np.concatenate([np.zeros(norm_test_samples.shape[0]), np.ones(norm_test_samples.shape[0])])
    return X_train, X_test, y_test



