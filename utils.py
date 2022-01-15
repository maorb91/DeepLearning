from measurements import metrics ,get_best_threshold_value
from load_dataset import load_dataset
from vae import VAE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from measurements import pick_best_model_from_csv


def save_best_model_to_csv(models, file_name):

    # save models results to a csv file
    columns = ["model_name", "model_id", "best_threshold", "F1", "AUC", "Accuracy", "anomalies_percentage"]
    rows = []
    x_train, X_test, Y_test = load_dataset(train_precent=0.8, seed=42)
    X_val, x_test, y_val, y_test = train_test_split(X_test, Y_test, test_size=0.2, random_state=42)

    # for each model calc the metrics

    for model in models :
        path = "model\\" + model
        vae = VAE().load(path)
        x_val_pred, _ = vae.reconstruct(X_val)

        # calc metrics :
        metrics_df = metrics(X_val, x_val_pred, y_val)

        # get best threshold level
        best_th = get_best_threshold_value(metrics_df, ["Accuracy"])

        x_pred, _ = vae.reconstruct(x_test)
        anomalies = np.mean(abs(x_test - x_pred), axis=1) > best_th

        anomalies_percentage = np.count_nonzero(anomalies) / len(anomalies)
        # get the index of the best result
        max_index = metrics_df["Accuracy"].idxmax()

        # save result as df row
        row = [model, models[model], best_th, metrics_df.iloc[max_index]["F1"], metrics_df.iloc[max_index]["AUC"],
               metrics_df.iloc[max_index]["Accuracy"],anomalies_percentage]

        rows.append(row)
    # convert df to csv
    best = pd.DataFrame(rows, columns=columns)
    best.head(15)
    best.to_csv(file_name, encoding='utf-8', index=False)
    print("saved result to csv file ")


def load_model(model_name):
    vae = VAE().load(model_name)
    return vae


if __name__ == "__main__":

    models = {'encoder_size_2_decoder_size_2_latent_size_2_beta_1_reconstruction_loss_1': 1,
              'encoder_size_2_decoder_size_2_latent_size_5_beta_1_reconstruction_loss_1': 2,
              'encoder_size_2_decoder_size_2_latent_size_10_beta_1_reconstruction_loss_1': 3,
              'encoder_size_3_decoder_size_3_latent_size_2_beta_1_reconstruction_loss_1': 4,
              'encoder_size_3_decoder_size_3_latent_size_5_beta_1_reconstruction_loss_1': 5,
              'encoder_size_3_decoder_size_3_latent_size_10_beta_1_reconstruction_loss_1': 6,
              'encoder_size_4_decoder_size_4_latent_size_2_beta_1_reconstruction_loss_1': 7,
              'encoder_size_4_decoder_size_4_latent_size_5_beta_1_reconstruction_loss_1': 8,
              'encoder_size_4_decoder_size_4_latent_size_10_beta_1_reconstruction_loss_1': 9
              }
    base_file_name = 'results\\base_VAE_Model.csv'

    save_best_model_to_csv(models, base_file_name)

    models = {
        'encoder_size_2_decoder_size_2_latent_size_2_beta_5_reconstruction_loss_1': 10,
        'encoder_size_2_decoder_size_2_latent_size_5_beta_5_reconstruction_loss_1': 11,
        'encoder_size_2_decoder_size_2_latent_size_10_beta_5_reconstruction_loss_1': 12,
        'encoder_size_3_decoder_size_3_latent_size_2_beta_5_reconstruction_loss_1': 13,
        'encoder_size_3_decoder_size_3_latent_size_5_beta_5_reconstruction_loss_1': 14,
        'encoder_size_3_decoder_size_3_latent_size_10_beta_5_reconstruction_loss_1': 15,
        'encoder_size_4_decoder_size_4_latent_size_2_beta_5_reconstruction_loss_1': 16,
        'encoder_size_4_decoder_size_4_latent_size_5_beta_5_reconstruction_loss_1': 17,
        'encoder_size_4_decoder_size_4_latent_size_10_beta_5_reconstruction_loss_1': 18
    }

    beta_file_name = 'results\\beta_VAE_Models_result.csv'
    save_best_model_to_csv(models, beta_file_name)

    models = {'encoder_size_2_decoder_size_2_latent_size_2_beta_1_reconstruction_loss_100': 19,
              'encoder_size_2_decoder_size_2_latent_size_5_beta_1_reconstruction_loss_100': 20,
              'encoder_size_2_decoder_size_2_latent_size_10_beta_1_reconstruction_loss_100': 21,
              'encoder_size_3_decoder_size_3_latent_size_2_beta_1_reconstruction_loss_100': 22,
              'encoder_size_3_decoder_size_3_latent_size_5_beta_1_reconstruction_loss_100': 23,
              'encoder_size_3_decoder_size_3_latent_size_10_beta_1_reconstruction_loss_100': 24,
              'encoder_size_4_decoder_size_4_latent_size_2_beta_1_reconstruction_loss_100': 25,
              'encoder_size_4_decoder_size_4_latent_size_5_beta_1_reconstruction_loss_100': 26,
              'encoder_size_4_decoder_size_4_latent_size_10_beta_1_reconstruction_loss_100': 27
              }

    reconstruction_loss_file_name = 'results\\reconstruction_loss_VAE_Models_result.csv'
    save_best_model_to_csv(models, reconstruction_loss_file_name)
    pick_best_model_from_csv([base_file_name,beta_file_name,reconstruction_loss_file_name])