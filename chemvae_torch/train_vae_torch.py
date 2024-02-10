import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from functools import partial


# Define the models, loss functions, etc.
# You need to define your PyTorch models and other necessary components here.
# This includes defining the encoder, decoder, property predictor, loss functions,
# and any other required components.

# For demonstration purposes, let's assume you have defined your PyTorch models as follows:

# encoder = YourEncoderModel()
# decoder = YourDecoderModel()
# property_predictor = YourPropertyPredictorModel()

def main_property_run(params):
    start_time = time.time()

    # Load data
    X_train, X_test, Y_train, Y_test = vectorize_data(params)

    # Load full models
    encoder, decoder, property_predictor = load_models(params)

    # Define loss functions
    criterion_ae = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    # Define optimizer
    if params['optim'] == 'adam':
        optimizer = optim.Adam(encoder.parameters(), lr=params['lr'], betas=(params['momentum'], 0.999))
    elif params['optim'] == 'rmsprop':
        optimizer = optim.RMSprop(encoder.parameters(), lr=params['lr'], momentum=params['momentum'])
    elif params['optim'] == 'sgd':
        optimizer = optim.SGD(encoder.parameters(), lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplementedError("Please define valid optimizer")

    # Prepare data loaders (assuming X_train, X_test, Y_train, Y_test are PyTorch tensors)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    # Train the model
    for epoch in range(params['prev_epochs'], params['epochs']):
        encoder.train()
        decoder.train()
        property_predictor.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            output_ae, _, _ = encoder(data)
            recon_loss = criterion_ae(output_ae, data)

            # Compute KL divergence loss
            kl_loss = kl_divergence_loss(encoder)

            # Property predictor loss
            output_prop = property_predictor(data)
            prop_loss = criterion_ae(output_prop, target)

            # Total loss
            loss = recon_loss + params['kl_loss_weight'] * kl_loss + prop_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Print some metrics or do logging
        if epoch % params['log_interval'] == 0:
            print(f"Epoch {epoch}: Loss: {loss.item()}, Recon Loss: {recon_loss.item()}, KL Loss: {kl_loss.item()}, Prop Loss: {prop_loss.item()}")

    # Save the models
    torch.save(encoder.state_dict(), params['encoder_weights_file'])
    torch.save(decoder.state_dict(), params['decoder_weights_file'])
    torch.save(property_predictor.state_dict(), params['prop_pred_weights_file'])

    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')

    return

# Define your kl divergence loss function
def kl_divergence_loss(encoder):
    # Your implementation of KL divergence loss here
    pass
