import torch
import torch.nn as nn
import torch.nn.init as init

# Define an Autoencoder model with Glorot initialization
# Note the AE architecture must be modified for each specific dataset.
class AutoEncoder(nn.Module):
    def __init__(self, layer_sizes):
        """
        A class to construct an autoencoder-like neural network model consisting
        of an encoder-decoder structure. This architecture builds both the encoder
        and decoder based on a configurable list of layer sizes, and uses the
        LeakyReLU activation function between layers, where appropriate.
        Weights of the model are initialized upon instantiation.

        :param layer_sizes: A list that defines the dimensions of each layer in the autoencoder
            structure. The first element is the input size, the last is the latent or bottleneck
            size, and the intermediate elements are the hidden layer sizes. Used to configure
            both the encoder and decoder parts of the autoencoder.
        :type layer_sizes: list[int]
        """
        super().__init__()

        # Build the encoder
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                # Add a LeakyReLU activation function between layers except for the last one
                encoder_layers.append(nn.LeakyReLU(0.1))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build the decoder
        decoder_layers = []
        for i in range(len(layer_sizes) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i - 1]))
            if i > 1:
                # Add a LeakyReLU activation function between layers except for the first one
                decoder_layers.append(nn.LeakyReLU(0.1))
        self.decoder = nn.Sequential(*decoder_layers)

        self.latent_dim = layer_sizes[-1]
        self._initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    fan_in, fan_out = module.weight.size(1), module.weight.size(0)
                    std = float(torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out))).item())
                    with torch.no_grad():
                        module.bias.uniform_(-std, std)

# Define an D-SVDD model
class DeepSVDD(nn.Module):
    def __init__(self, enc_arch):
        super().__init__()
        self.encoder = enc_arch

        # Remove bias terms
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.bias = None
            elif isinstance(module, nn.Conv2d):
                module.bias = None

        # Get the latent space dimension
        last_layer = list(self.encoder.modules())[-1]
        if isinstance(last_layer, nn.Linear):
            self.latent_dim = last_layer.out_features
        elif isinstance(last_layer, nn.Conv2d):
            self.latent_dim = last_layer.out_channels

    def forward(self, x):
        return self.encoder(x)

    def set_init_weights(self, ae_model):
        """
        Copy the pretrained autoencoder weights to the Deep SVDD model.

        :param ae_model: Pytorch model of the pretrained autoencoder.
        """

        enc_state_dict = {}
        for name, param in ae_model.state_dict().items():
            # Copy only the weights
            if 'encoder' in name and 'weight' in name:
                enc_state_dict[name.replace('encoder.', '')] = param

        # Load the copied weights into the model
        self.encoder.load_state_dict(enc_state_dict, strict=False)