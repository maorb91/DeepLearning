import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score
import pandas as pd


def plot_latent_space(model, x):
    if model.latent_space_dim == 2:
        latent_representations = model.encoder.predict(x)
        plt.figure(figsize=(12, 10))
        plt.scatter(latent_representations[:, 0], latent_representations[:, 1])
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.title('Latent Space representation')
        plt.show()
    print("")


def plot_label_latent_space(model, x,y):
    if model.latent_space_dim == 2 :
        latent_representations = model.encoder.predict(x)
        plt.figure(figsize=(12, 10))
        sns.scatterplot(latent_representations[:, 0], latent_representations[:, 1], hue=y, palette=['red', 'blue'] ,alpha = 0.2)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.title('Validation Data Latent Space representation')
        plt.show()
    print("")


def plot_reconstruction_error(model, x, label = None):
    # plot the reconstruction error of data x
    reconstructed_inputs, _ = model.reconstruct(x)
    reconstructed_error = np.mean(abs(x - reconstructed_inputs),axis=1)
    plt.figure()
    # if using labels plot the reconstruction error in different colors
    if label is not None:
        mask_norm = label == 1  # create mask of benign
        mask_att = label == 0
        plt.hist(reconstructed_error[np.where(mask_norm == 1)], color='b',
                 alpha=0.5, label='normal network connection')
        plt.hist(reconstructed_error[np.where(mask_att == 1)], color='r',
                 alpha=0.5, label='abnormal  network connection')
    else:
        plt.hist(reconstructed_error,label='train data')
    plt.ylabel('Count')
    plt.xlabel('reconstruction error')
    plt.title('Validation Data reconstruction error')
    plt.legend()
    plt.show()


def get_reconstruction_error(model, x):
    # get reconstruction error of x
    reconstructed_inputs, _ = model.reconstruct(x)
    reconstructed_error = np.mean(abs(x - reconstructed_inputs), axis=1)
    return np.quantile(reconstructed_error, 0.99)



def metrics(x, x_pred, y):
    # calc different metrics on data for each threshold value
    columns = ["threshold", "F1", "Precision", "Recall", "AUC", "TN", "FN", "TP", "FP", "Accuracy"]
    rows = []
    th = np.linspace(0, 0.15, num=50)
    for t in th:
        anomalies = np.mean(abs(x - x_pred), axis=1) > t
        CM = confusion_matrix(y, anomalies)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        f1 = f1_score(y,anomalies)
        precision = precision_score(y, anomalies)
        recall=recall_score(y, anomalies)
        roc = roc_auc_score(y, anomalies)
        accuracy = accuracy_score(y,anomalies)
        row = [t, f1, precision, recall, roc, TN, FN, TP, FP, accuracy]
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def get_best_threshold_value(result_df, columns=[]):
    # get the best threshold value by the metrics in columns
    if len(columns) > 0:
        for column in columns:
            max_index = result_df[column].idxmax()
            print(f"max value of {column} is {result_df.iloc[max_index][column]}")
            print(f"Th is {result_df.iloc[max_index]['threshold']}")
    return result_df.iloc[max_index]["threshold"]


def pick_best_model_from_csv(file_names):
    # read all csv file in file names
    # pick the best model according to Accuracy
    files_dfs = []
    for file_name in file_names:
        df = pd.read_csv(file_name)
        files_dfs.append(df)
    plt.figure(figsize=(12, 10))

    model_df = pd.concat(files_dfs, ignore_index=True)
    model_df.sort_values(by=['Accuracy'])
    max_index = model_df['Accuracy'].idxmax()
    plt.subplot(1, 1, 1)
    sns.barplot(x=model_df['model_id'],y=model_df['Accuracy'],data=model_df)
    plt.title(f"Model Accuracy Score\nbest Model is model : {model_df.iloc[max_index]['model_id']} - {model_df.iloc[max_index]['model_name']}\nAccuracy = {model_df.iloc[max_index]['Accuracy']} \nAUC = {model_df.iloc[max_index]['AUC']}")
    plt.show()
    return model_df.iloc[max_index]['model_name']
