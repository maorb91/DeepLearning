import matplotlib.pyplot as plt
from vae import VAE
from load_dataset import load_dataset
import numpy as np
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 10


def load_data():
    x_train, x_test, y_train = load_dataset(train_precent=0.8, seed=42)
    return x_train,x_test, y_train


def train(x_train, learning_rate,
          batch_size, epochs,
          encoder_dense_layers, decoder_dense_layers,
          latent_space_dim, beta =1,
          reconstruction_loss_weight =1):

    variational_autoencoder = VAE(
        input_size=x_train.shape[1],
        encoder_dense_layers=encoder_dense_layers,
        decoder_dense_layers=decoder_dense_layers,
        latent_space_dim=latent_space_dim,
        beta=beta,
        reconstruction_loss_weight=reconstruction_loss_weight
    )
    variational_autoencoder.summary()

    variational_autoencoder.compile(learning_rate)
    variational_autoencoder.train(x_train, batch_size, epochs)
    return variational_autoencoder


if __name__ == "__main__":
    x_train, _, _ = load_data()
    vautoencoder = train(x_train,
                         LEARNING_RATE,
                         BATCH_SIZE,
                         EPOCHS,
                         encoder_dense_layers =(80, 40),
                         decoder_dense_layers =(40, 80),
                         latent_space_dim =2,
                         beta =1,
                         reconstruction_loss_weight = 1)
    print("end training")
    print("saving  model training")
    vautoencoder.save()
    print("end...")

