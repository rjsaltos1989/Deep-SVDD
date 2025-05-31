import torch
import torch.nn as nn
import torch.nn.init as init

# Define an Autoencoder model with Glorot initialization
class AutoEncoder(nn.Module):
    def __init__(self, input_size,latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size,10),
            nn.LeakyReLU(0.1),
            nn.Linear(10, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 4),
            nn.LeakyReLU(0.1),
            nn.Linear(4, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.LeakyReLU(0.1),
            nn.Linear(4, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 10),
            nn.LeakyReLU(0.1),
            nn.Linear(10, input_size)
        )

        self.latent_dim = latent_dim
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