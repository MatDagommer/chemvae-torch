"""PyTorch models for the stability VAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import GRUCell


# Custom BatchNorm1d layer that has eps=1e-3 and does not use Bessel's correction (Keras' defaults)
class CustomBatchNorm1d(nn.Module):
    """Custom BatchNorm1d layer that has eps=1e-3 and does not use Bessel's correction (Keras' defaults)."""

    def __init__(self, num_features, eps=1e-3, momentum=0.1):
        """
        Init Custom BatchNorm1d layer that has eps=1e-3 and does not use Bessel's correction (Keras' defaults).

        :param num_features: number of features
        :param eps: epsilon
        :param momentum: momentum
        """
        super(CustomBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(1, num_features).fill_(0.1))
        self.bias = nn.Parameter(torch.Tensor(1, num_features).fill_(0))
        self.register_buffer("running_mean", torch.zeros(1, num_features))
        self.register_buffer("running_var", torch.ones(1, num_features))

    def forward(self, input):
        """
        Forward pass.

        :param input: input tensor
        :return: output tensor
        """
        # Expand dimensions of running mean and var to match input tensor
        if len(input.size()) > 2:
            running_mean = self.running_mean.unsqueeze(-1)
            running_var = self.running_var.unsqueeze(-1)
            weight = self.weight.unsqueeze(-1)
            bias = self.bias.unsqueeze(-1)
        else:
            running_mean = self.running_mean
            running_var = self.running_var
            weight = self.weight
            bias = self.bias

        X_hat = (input - running_mean) / torch.sqrt(running_var + self.eps)
        y = weight * X_hat + bias
        return y


class EncoderModel(nn.Module):
    """
    Encoder model.

    :param params: hyperparameters
    """

    def __init__(self, params):
        """
        Init Encoder model.

        :param params: hyperparameters
        """
        super(EncoderModel, self).__init__()
        self.params = params

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()

        # Convolution layers
        in_channels = params["NCHARS"]
        out_channels = int(params["conv_dim_depth"] * params["conv_d_growth_factor"])
        kernel_size = int(params["conv_dim_width"] * params["conv_w_growth_factor"])
        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv_layers.append(conv_layer)
        if params["batchnorm_conv"]:
            # norm_layer = nn.BatchNorm1d(out_channels)
            norm_layer = CustomBatchNorm1d(out_channels)
            self.conv_norm_layers.append(norm_layer)

        in_channels = out_channels

        for j in range(1, params["conv_depth"] - 1):
            out_channels = int(params["conv_dim_depth"] * params["conv_d_growth_factor"] ** j)
            kernel_size = int(params["conv_dim_width"] * params["conv_w_growth_factor"] ** j)
            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size)
            self.conv_layers.append(conv_layer)

            if params["batchnorm_conv"]:
                # norm_layer = nn.BatchNorm1d(out_channels)
                norm_layer = CustomBatchNorm1d(out_channels)
                self.conv_norm_layers.append(norm_layer)

            in_channels = out_channels

        self.flatten = nn.Flatten()

        # Middle layers
        if params["middle_layer"] > 0:
            self.middle_layers = nn.ModuleList()
            self.dropout_layers = nn.ModuleList()
            self.middle_norm_layers = nn.ModuleList()

            # TODO: find a way to calculate in_features automatically
            in_features = out_channels * 94

            for i in range(1, params["middle_layer"] + 1):
                out_features = int(params["hidden_dim"] * params["hg_growth_factor"] ** (params["middle_layer"] - i))
                middle_layer = nn.Linear(in_features, out_features)
                self.middle_layers.append(middle_layer)

                if params["dropout_rate_mid"] > 0:
                    dropout_layer = nn.Dropout(params["dropout_rate_mid"])
                    self.dropout_layers.append(dropout_layer)

                if params["batchnorm_mid"]:
                    # norm_layer = nn.BatchNorm1d(out_features)
                    norm_layer = CustomBatchNorm1d(out_features)
                    self.middle_norm_layers.append(norm_layer)

                in_features = out_features

        # output has dim = hidden_dim = 100 (hyperparameters.py)
        self.z_mean = nn.Linear(in_features, params["hidden_dim"])
        self.z_logvar = nn.Linear(in_features, params["hidden_dim"])
        

    def forward(self, x):
        """
        Forward pass.

        :param x: input tensor
        :return: mean and last encoding layer for std dev sampling
        """

        print("4: ", type(x))

        # Transpose input
        x = x.transpose(2, 1)

        # Convolution layers
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = torch.tanh(x)  # activation
            if self.params["batchnorm_conv"]:
                x = self.conv_norm_layers[i](x)

        x = x.transpose(2, 1)
        x = self.flatten(x)
        # print("x.size(): ", x.size())

        # Middle layers
        if self.params["middle_layer"] > 0:
            for i in range(len(self.middle_layers)):
                # print("TEST: ", i)
                x = self.middle_layers[i](x)
                x = torch.tanh(x)
                if self.params["dropout_rate_mid"] > 0:
                    x = self.dropout_layers[i](x)
                if self.params["batchnorm_mid"]:
                    x = self.middle_norm_layers[i](x)

        # output has dim = hidden_dim = 100 (hyperparameters.py)
        z_mean = self.z_mean(x)
        z_logvar = self.z_logvar(x)

        # return both mean and encoder output
        # return z_mean, x
        return z_mean, z_logvar


class DecoderModel(nn.Module):
    """
    Decoder model.

    :param params: hyperparameters
    """

    def __init__(self, params):
        """
        Init Decoder model.

        :param params: hyperparameters
        """
        super(DecoderModel, self).__init__()
        self.params = params

        self.z = nn.Sequential(
            nn.Linear(params["hidden_dim"], int(params["hidden_dim"])),
            nn.Dropout(params["dropout_rate_mid"]) if params["dropout_rate_mid"] > 0 else nn.Identity(),
            CustomBatchNorm1d(int(params["hidden_dim"])) if params["batchnorm_mid"] else nn.Identity(),
        )

        for i in range(1, params["middle_layer"]):
            self.z.add_module(
                "decoder_dense{}".format(i),
                nn.Sequential(
                    nn.Linear(
                        int(params["hidden_dim"] * params["hg_growth_factor"] ** (i)),
                        int(params["hidden_dim"] * params["hg_growth_factor"] ** (i)),
                    ),
                    nn.Dropout(params["dropout_rate_mid"]) if params["dropout_rate_mid"] > 0 else nn.Identity(),
                    CustomBatchNorm1d(int(params["hidden_dim"] * params["hg_growth_factor"] ** (i)))
                    if params["batchnorm_mid"]
                    else nn.Identity(),
                ),
            )

        if params["gru_depth"] > 1:
            self.x_dec = nn.Sequential(nn.GRU(params["hidden_dim"], params["recurrent_dim"], batch_first=True))
            for i in range(1, params["gru_depth"] - 1):
                self.x_dec.add_module(
                    "gru_{}".format(i), nn.GRU(params["recurrent_dim"], params["recurrent_dim"], batch_first=True)
                )

        self.x_out = CustomGRU(params["recurrent_dim"], params["NCHARS"], 1, device=params["device"])
        # self.x_out = nn.GRU(params["recurrent_dim"], params["NCHARS"], 1)

    def forward(self, z_in, targets=None):
        """
        Forward pass.

        :param z_in: input tensor
        :return: output tensor
        """
        z = self.z(z_in)
        z_reshaped = z.unsqueeze(1)
        # Repeat z along the second dimension to get shape (batch_size, MAX_LEN, hidden_dim)
        z_reps = z_reshaped.repeat(1, self.params["MAX_LEN"], 1)

        if hasattr(self, "x_dec"):
            for i in range(len(self.x_dec)):
                z_reps, _ = self.x_dec[i](z_reps)
        x_dec = z_reps

        if self.training and targets is None:
            raise KeyError("The decoder is in training mode, but no targets were provided.")

        if self.training:
            # teacher forcing with the targets
            x_out, _ = self.x_out.forward(x_dec, targets=targets)
            # x_out, _ = self.x_out.forward(x_dec)
        else:
            x_out, _ = self.x_out.forward(x_dec)

        return x_out


class PropertyPredictorModel(nn.Module):
    """
    Property predictor model.

    :param params: hyperparameters
    """

    def __init__(self, params):
        """
        Property predictor model.

        :param params: hyperparameters
        :raises ValueError: if neither regression nor logistic tasks are specified
        """
        super(PropertyPredictorModel, self).__init__()

        if ("reg_prop_tasks" not in params) and ("logit_prop_tasks" not in params):
            raise ValueError("You must specify either regression tasks and/or logistic tasks for property prediction")

        self.ls_in = nn.Linear(params["hidden_dim"], params["prop_hidden_dim"])
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(params["prop_pred_dropout"])
        self.hidden_layers = nn.ModuleList()

        if params["prop_pred_depth"] > 1:
            for p_i in range(1, params["prop_pred_depth"]):
                hidden_layer = nn.Linear(
                    int(params["prop_hidden_dim"] * params["prop_growth_factor"] ** (p_i - 1)),
                    int(params["prop_hidden_dim"] * params["prop_growth_factor"] ** p_i),
                )
                self.hidden_layers.append(hidden_layer)
                # self.hidden_layers.append(self.activation)

                if params["prop_pred_dropout"] > 0:
                    dropout_layer = nn.Dropout(params["prop_pred_dropout"])
                    self.hidden_layers.append(dropout_layer)

                if params["prop_batchnorm"]:
                    # norm_layer = nn.BatchNorm1d(int(params["prop_hidden_dim"] * params["prop_growth_factor"] ** p_i))
                    norm_layer = CustomBatchNorm1d(int(params["prop_hidden_dim"] * params["prop_growth_factor"] ** p_i))
                    self.hidden_layers.append(norm_layer)

        # Regression tasks
        if ("reg_prop_tasks" in params) and (len(params["reg_prop_tasks"]) > 0):
            self.reg_prop_pred = nn.Linear(
                int(params["prop_hidden_dim"] * params["prop_growth_factor"] ** (params["prop_pred_depth"] - 1)),
                len(params["reg_prop_tasks"]),
            )

        # Logistic tasks
        if ("logit_prop_tasks" in params) and (len(params["logit_prop_tasks"]) > 0):
            self.logit_prop_pred = nn.Linear(
                int(params["prop_hidden_dim"] * params["prop_growth_factor"] ** (params["prop_pred_depth"] - 1)),
                len(params["logit_prop_tasks"]),
            )
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        out = self.ls_in(x)
        # out = self.activation(out)
        out = torch.tanh(out)
        out = self.dropout(out)

        # for hidden_layer in self.hidden_layers:
        out = self.hidden_layers[0](out)
        out = torch.tanh(out)
        out = self.hidden_layers[1](out)
        out = self.hidden_layers[2](out)
        out = self.hidden_layers[3](out)
        out = torch.tanh(out)
        out = self.hidden_layers[4](out)
        out = self.hidden_layers[5](out)

        # Regression tasks
        if hasattr(self, "reg_prop_pred"):
            reg_prop_pred = self.reg_prop_pred(out)

        # Logistic tasks
        if hasattr(self, "logit_prop_pred"):
            logit_prop_pred = self.logit_prop_pred(out)
            logit_prop_pred = self.sigmoid(logit_prop_pred)

        # Both regression and logistic
        if hasattr(self, "reg_prop_pred") and hasattr(self, "logit_prop_pred"):
            return reg_prop_pred, logit_prop_pred

        # Regression only scenario
        elif hasattr(self, "reg_prop_pred"):
            return reg_prop_pred

        # Logistic only scenario
        else:
            return logit_prop_pred


class CustomGRUCell(GRUCell):
    """
    Custom GRU cell that uses softmax instead of sigmoid for the new gate.

    This is to ensure that the new gate is always positive and sums to 1.

    :param input_size: input size
    :param hidden_size: hidden size
    :param bias: whether to use bias
    """

    def __init__(self, input_size, hidden_size, bias=True, device=None):
        """
        Init Custom GRU cell that uses softmax instead of sigmoid for the new gate.

        :param input_size: input size
        :param hidden_size: hidden size
        :param bias: whether to use bias
        """
        super(CustomGRUCell, self).__init__(input_size, hidden_size, bias)
        # weights for teacher forcing
        self.weight_tf = torch.randn(hidden_size, hidden_size, requires_grad=True).to(device)
    
    def forward(self, input, hx=None, prev_target=None):
        """
        Forward pass.

        :param input: input tensor
        :param hx: hidden state
        :return: output tensor
        """
        # self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        if prev_target is None:
            prev_target = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        x_gates = F.linear(input, self.weight_ih) # self.weight_ih = concat(W_r, W_z, W_h)
        h_gates = F.linear(hx, self.weight_hh) # self.weight_ih = concat(U_r, U_z, U_h)

        x_r, x_z, x_h = x_gates.chunk(3, 1) # returns W_r @ x_t, W_z @ x_t, W_h @ x_t 
        h_r, h_z, h_h = h_gates.chunk(3, 1) # returns U_r @ h_tm1, U_z @ h_tm1, U_h @ h_tm1 

        reset_gate = torch.sigmoid(x_r + h_r + self.bias_ih[:self.hidden_size])
        update_gate = torch.sigmoid(x_z + h_z + self.bias_ih[self.hidden_size:2*self.hidden_size])

        new_gate = F.softmax(
            x_h
            + F.linear(reset_gate * hx, self.weight_hh[2*self.hidden_size:])
            # + reset_gate * h_h # this line seems to yield better results, although it is incorrect
            + F.linear(reset_gate * prev_target, self.weight_tf)
            + self.bias_ih[2*self.hidden_size:],
            dim=1
        )

        hy = (1. - update_gate) * new_gate + update_gate * hx

        return hy

class CustomGRU(torch.nn.Module):
    """
    Custom GRU layer that uses softmax instead of sigmoid for the new gate.

    :param input_size: input size
    :param hidden_size: hidden size
    :param num_layers: number of layers
    """

    def __init__(self, input_size, hidden_size, num_layers, device):
        """
        Init Custom GRU layer that uses softmax instead of sigmoid for the new gate.

        :param input_size: input size
        :param hidden_size: hidden size
        :param num_layers: number of layers
        """
        super(CustomGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = CustomGRUCell(input_size, hidden_size, device=device)
        # self.cell = GRUCell(input_size, hidden_size)
        # for _ in range(1, num_layers):
        #     self.cells.append(CustomGRUCell(hidden_size, hidden_size))

    # def forward(self, inputs, hx=None):
    #     """
    #     Forward pass.

    #     :param inputs: input tensor
    #     :param hx: hidden state
    #     :return: output tensor
    #     """
    #     if hx is None:
    #         hx = torch.zeros(inputs.size(0), inputs.size(1) + 1, self.hidden_size, device=inputs.device)
    #     # inputs: batch_size x seq_len x input_size
    #     outputs = []
    #     for i in range(inputs.size(1)):
    #         hx[:, i + 1] = self.cell(inputs[:, i], hx[:, i])
    #         outputs.append(hx[:, i + 1])

    #     return torch.stack(outputs, dim=1), hx
        
    def sample_from_probabilities(self, prob_tensor, device):
        batch_size, hidden_dim = prob_tensor.size()
        
        # Sample indices from the distributions
        indices = torch.multinomial(prob_tensor, num_samples=1).view(-1).to(device)
        
        # Create the one-hot encoded tensor
        one_hot = torch.zeros(batch_size, hidden_dim).to(device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1).to(device)
        
        return one_hot
    
    def forward(self, inputs, targets=None, hx=None):
        """
        Forward pass.

        :param inputs: input tensor
        :param hx: hidden state
        :return: output tensor
        """
        if hx is None:
            # hx = torch.zeros(inputs.size(0), inputs.size(1) + 1, self.hidden_size, device=inputs.device)
            hx = torch.zeros(inputs.size(0), 1, self.hidden_size, device=inputs.device)
            # hx = torch.ones(inputs.size(0), 1, self.hidden_size, device=inputs.device) / self.hidden_size
        # inputs: batch_size x seq_len x input_size
        outputs = []
        # print(inputs.size())
        # Creating a tensor that contains the targets + zeros at the beginning 
        # (first iteration has no teacher forcing)
        if targets is not None:
            targets_and_zeros = torch.zeros(inputs.size(0), inputs.size(1) + 1, self.hidden_size, device=inputs.device)
            targets_and_zeros[:, 1:, :] = targets
            # print(targets_and_zeros[:, 0, :])
        
        for i in range(inputs.size(1)):
            if self.training:
                # Use teacher forcing by replacing the computed hidden state with the actual previous target
                next_hidden = self.cell(inputs[:, i], hx=hx[:, i], prev_target=targets_and_zeros[:, i]).to(inputs.device)
                # next_hidden = self.cell(inputs[:, i], hx=hx[:, i])
            else:
                # predict next hidden state
                next_hidden = self.cell(inputs[:, i], hx=hx[:, i]).to(inputs.device)
                # adding a sampling step to retrieve a one-hot
                # next_hidden = F.softmax(next_hidden, dim=1)
                # next_hidden = torch.max(next_hidden, dim=1)
                # row_sums = next_hidden.sum(dim=0)
                # Divide each row by its sum using broadcasting
                # result = next_hidden / row_sums[:, None]
                # next_hidden = self.sample_from_probabilities(next_hidden, inputs.device)
            hx = torch.cat((hx, next_hidden.unsqueeze(1)), dim=1)

            outputs.append(next_hidden)

        return torch.stack(outputs, dim=1).to(inputs.device), hx


class AE_PP_Model(nn.Module):
    """
    Variational Autoencoder with property prediction (QED, SAS, logP).
    """
    def __init__(self, encoder, decoder, params, device, property_predictor=None):
        
        super(AE_PP_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.loss_weights = params["model_loss_weights"]
        self.hidden_dim = params["hidden_dim"]
        self.use_mu = params["use_mu"]
        self.do_prop_pred = params["do_prop_pred"]

        if self.do_prop_pred:
            self.property_predictor = property_predictor

        self.device = device

        # similar to the layer found in models.variational_layers (original code)
        # self.logvar_layer = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        # self.batch_norm_vae = CustomBatchNorm1d(self.hidden_dim)

    def reparameterize(self, mu, logvar, kl_loss_weight):
        std = torch.exp(0.5 * logvar).to(self.device)
        eps = torch.randn_like(std).to(self.device)
        z = mu + eps * std #  * kl_loss_weight
        return z

    def forward(self, x, kl_loss_weight=None):
        # Encode input - returns mean and encoder output
        # mu, encoder_output = self.encoder(x)

        print("3: ", type(x))

        mu, logvar = self.encoder(x)
        
        # Retrieving property prediction
        if self.do_prop_pred:
            prediction = self.property_predictor(mu)

        # Compute log variance from the encoder's output
        # logvar = self.logvar_layer(encoder_output)
        
        if kl_loss_weight is None:
            kl_loss_weight = 0

        # Reparameterization trick to sample from the latent space
        z = self.reparameterize(mu, logvar, kl_loss_weight)

        # # batchnormalization
        # z = self.batch_norm_vae(z)
        
        # Decode the latent variable
        if self.use_mu:
            # decoding using the mean only (no stochasticity)
            reconstruction = self.decoder(mu, x)
        else:
            # with sampling (classic VAE training)
            reconstruction = self.decoder(z, x)

        if self.do_prop_pred:
            return reconstruction, prediction, mu, logvar
        else:
            return reconstruction, mu, logvar

    def loss_function(self, reconstruction, mu, logvar, x, kl_loss_weight, y=None, prediction=None):

        # Compute reconstruction loss
        reconstruction_criterion = nn.CrossEntropyLoss()
        
        # reshaping to 2D tensors
        reconstruction = reconstruction.view(reconstruction.size(0), -1)
        x = x.view(x.size(0), -1)
        reconstruction_loss = reconstruction_criterion(reconstruction, x)

        # Compute KL loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Compute prediction loss
        if prediction is not None:
            prediction_criterion = nn.MSELoss()
            prediction_loss = prediction_criterion(prediction, y)

            # Total loss
            total_loss = reconstruction_loss * self.loss_weights["reconstruction_loss"] \
                        + kl_loss * kl_loss_weight * self.loss_weights["ae_loss"] \
                        + prediction_loss * self.loss_weights["prediction_loss"]
            
            return total_loss, reconstruction_loss, kl_loss, prediction_loss
        
        else:
            # Total loss
            total_loss = reconstruction_loss * self.loss_weights["reconstruction_loss"] \
                        + kl_loss * kl_loss_weight

            return total_loss, reconstruction_loss, kl_loss
        
