from tensorflow.keras import Model, callbacks
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import datetime
import pickle
tf.compat.v1.disable_eager_execution()


class VAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_size=0,
                 encoder_dense_layers=(),
                 decoder_dense_layers=(),
                 latent_space_dim=0,
                 beta=1,
                 reconstruction_loss_weight=1):
        self.input_size = input_size
        self.encoder_dense_layers = encoder_dense_layers
        self.decoder_dense_layers = decoder_dense_layers
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.beta = beta

        self.encoder = None
        self.decoder = None
        self.model = None

        self._encoder_dense_layers = len(encoder_dense_layers)
        self._decoder_dense_layers = len(decoder_dense_layers)
        self._shape_before_bottleneck = None
        self._model_input = None
        self._callbacks = []
        # self._callbacks = self._get_callbacks() # uncomment this line to used tensorboard

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[self._calculate_reconstruction_loss,
                                    self._calculate_kl_loss])

    def train(self, x_train, batch_size, num_epochs):

        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True,
                       validation_split=0.2,
                       callbacks=self._callbacks)

    def save(self, save_folder="."):
        self._create_folder_if_not_exists(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def load(self, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def reconstruct(self, inputs):
        latent_representations = self.encoder.predict(inputs)
        reconstructed_inputs = self.decoder.predict(latent_representations)
        return reconstructed_inputs, latent_representations

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + self.beta * kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=1)
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
        return kl_loss

    def _create_folder_if_not_exists(self,folder):
        if not  os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_size,
            self.encoder_dense_layers,
            self.decoder_dense_layers,
            self.latent_space_dim,
            self.beta,
            self.reconstruction_loss_weight
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_vae()

    def _build_vae(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="vae")

    def _get_callbacks(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,  # How often to log histogram visualizations
            embeddings_freq=0,  # How often to log embedding visualizations
            update_freq="epoch",
        )

        return [tensorboard_callback]

    #  region Decoder
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_decoder_dense_layers(decoder_input)
        decoder_output = self._add_decoder_output(dense_layer)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_decoder_dense_layers(self, decoder_input):
        """Create all convolutional blocks in encoder."""
        x = decoder_input
        for layer_index in range(self._decoder_dense_layers):
            x = self._add_decoder_dense_layer(layer_index, x)
        return x

    def _add_decoder_dense_layer(self, layer_index, x):
        """Add a dense block to a graph of layers, consisting of
        dense + ReLU.
        """
        layer_number = layer_index + 1
        dense_layer = Dense(
            self.decoder_dense_layers[layer_index],
            activation="relu",
            name=f"decoder_dense_layer_{layer_number}"
        )
        return dense_layer(x)

    def _add_decoder_output(self, x):
        output_layer = Dense(
            self.input_size,
            activation="sigmoid",
            name="decoder_output"
        )
        return output_layer(x)
    #  endregion

    #  region Encoder

    def _build_encoder(self):
        encoder_input = self._add_encoder_input(self.input_size)
        dense_layers = self._add_dense_layers(encoder_input)
        bottleneck = self._add_bottleneck(dense_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self,input_size):
        return Input(shape=input_size, name="encoder_input")

    def _add_dense_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._encoder_dense_layers):
            x = self._add_encoder_dense_layer(layer_index, x)
        return x

    def _add_encoder_dense_layer(self, layer_index, x):
        """Add a dense block to a graph of layers, consisting of
        dense + ReLU.
        """
        layer_number = layer_index + 1
        dense_layer = Dense(
            self.encoder_dense_layers[layer_index],
            activation="relu",
            name=f"encoder_dense_layer_{layer_number}"
        )
        return dense_layer(x)

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Guassian sampling (Dense
        layer).
        """
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
                                      stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sample_point_from_normal_distribution,
                   name="z")([self.mu, self.log_variance])
        return x
    #  endregion


