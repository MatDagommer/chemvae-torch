from pathlib import Path
project_path = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(project_path))
print(project_path)

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from functools import partial
from chemvae_torch.models_torch import EncoderModel, DecoderModel, PropertyPredictorModel, AE_PP_Model
from chemvae_torch import hyperparameters
import argparse
import os
import pickle
import yaml
import chemvae_torch.mol_utils as mu
from chemvae_torch.utils_torch import vectorize_data, hot_to_smiles, schedule, plot_losses

# Define the models, loss functions, etc.
# You need to define your PyTorch models and other necessary components here.
# This includes defining the encoder, decoder, property predictor, loss functions,
# and any other required components.

# For demonstration purposes, let's assume you have defined your PyTorch models as follows:

# encoder = YourEncoderModel()
# decoder = YourDecoderModel()
# property_predictor = YourPropertyPredictorModel()

def train(params):
    start_time = time.time()
    device = params["device"]

    # Load data
    X_train, X_test, Y_train, Y_test = vectorize_data(params)

    # Keeping only regression property prediction targets
    Y_train = Y_train[0]
    Y_test = Y_test[0]

    # Convert data to torch tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    Y_train = torch.from_numpy(Y_train).float().to(device)
    Y_test = torch.from_numpy(Y_test).float().to(device)

    # Instantiate models
    encoder = EncoderModel(params)
    decoder = DecoderModel(params)
    property_predictor = PropertyPredictorModel(params)

    # Load pretrained parameters 
    if params["pretrained"]:
        encoder.load_state_dict(torch.load(params['pretrained_encoder_file']))
        # decoder.load_state_dict(torch.load(params['pretrained_decoder_file']))
        property_predictor.load_state_dict(torch.load(params['pretrained_predictor_file']))

    # Retrieving loss weights
    #NOTE: In original implementation, xent_loss_weight and kl_loss_weight are trainable
    ae_loss_weight = 1. - params['prop_pred_loss_weight']  # Autoencoder (Reconstruction + KL)
    xent_loss_weight = params['xent_loss_weight'] # Reconstruction (AE 1/2)
    kl_loss_weight = params['kl_loss_weight'] # KL (AE 2/2)
    prop_pred_loss_weight = params['prop_pred_loss_weight'] # Property Prediction
    
    model_loss_weights = {
                    'ae_loss': ae_loss_weight,
                    'reconstruction_loss': ae_loss_weight * xent_loss_weight,
                    'prediction_loss': prop_pred_loss_weight
                    }
    
    params["model_loss_weights"] = model_loss_weights
    
    vae = AE_PP_Model(encoder, decoder, params, device, property_predictor=property_predictor)

    # print(vae)

    # move models to GPU
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    property_predictor = property_predictor.to(device)
    vae = vae.to(device)

    print("TRAIN ALL: ", params["train_all"])
    
    if not params["train_all"]:
        # Freeze encoder
        for param in vae.encoder.parameters():
            param.requires_grad = False
        
        for param in vae.encoder.z_logvar.parameters():
            param.requires_grad = True

        for param in vae.encoder.z_mean.parameters():
            param.requires_grad = True
        
        # Freeze property predictor
        for param in vae.property_predictor.parameters():
            param.requires_grad = False
    
        # for param in vae.logvar_layer.parameters():
        #     param.requires_grad = True
    
    if params["train_logvar_only"]:
        # training logvar layer only (not training encoder, decoder, and prop pred)
        for param in vae.parameters():
            param.requires_grad = False
        
        for param in vae.encoder.z_logvar.parameters():
            param.requires_grad = True

        print("Training Decoder?: ", vae.decoder.training)

    # Define optimizer
    if params['optim'] == 'adam':
        optimizer = optim.Adam(vae.parameters(), lr=params['lr'], betas=(params['momentum'], 0.999))
        # optimizer = optim.Adam(vae.parameters())
    elif params['optim'] == 'rmsprop':
        optimizer = optim.RMSprop(vae.parameters(), lr=params['lr'], momentum=params['momentum'])
    elif params['optim'] == 'sgd':
        optimizer = optim.SGD(vae.parameters(), lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplementedError("Please define valid optimizer")

    # Prepare data loaders (assuming X_train, X_test, Y_train, Y_test are PyTorch tensors)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    test_batch, _ = next(iter(test_loader))
    test_sample = test_batch[0:1]

    train_batch, _ = next(iter(train_loader))
    train_sample = train_batch[0:1]

    # retrieving character encoding
    chars = yaml.safe_load(open(params["char_file"]))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    train_recon_losses = []
    train_kl_losses = []
    train_pred_losses = []

    batches_per_epoch = len(train_loader)

    # Train the model
    for epoch in range(params['prev_epochs'], params['epochs']):
        
        train_recon_loss = 0
        train_kl_loss = 0
        train_pred_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):

            vae.train()


            # update KL loss weight based on schedule
            kl_loss_weight = schedule(
                epoch,
                slope=params["anneal_sigmod_slope"],
                start=params["vae_annealer_start"],
                weight_orig=params["kl_loss_weight"],
                mode=params["schedule"]
            )

            # print("requires grad?: ", vae.encoder.z_logvar.weight.requires_grad)
            # print("logvar_layer weight: ", vae.encoder.z_logvar.weight[0])
            # print("logvar_layer weight grad: ", vae.encoder.z_logvar.weight[0].grad)
            # Forward pass
            reconstruction, mu, logvar, prediction = vae.forward(x, kl_loss_weight=kl_loss_weight)

            loss, recon_loss, kl_loss, pred_loss = vae.loss_function(
                reconstruction=reconstruction,
                mu=mu, 
                logvar=logvar, 
                x=x, 
                kl_loss_weight=kl_loss_weight,
                y=y,
                prediction=prediction,
            )

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print Losses
            # Format and print the strings with alignment
            print("Epoch {} Batch {}.".format(epoch, batch_idx))
            print(f"Losses - REC: {recon_loss.item()} KL: {kl_loss.item()} PRED: {pred_loss.item()} TOT: {loss.item()}")
            
            expected = hot_to_smiles(x.detach().cpu().numpy(), indices_char)[0]
            computed = hot_to_smiles(reconstruction.detach().cpu().numpy(), indices_char)[0]

            max_length = max(len(expected), len(computed))

            print(f"Target: {expected:<{max_length}}")
            print(f"Output: {computed:<{max_length}}")

            with open("pickle_file", "wb") as file:
                pickle.dump((x, reconstruction), file)

            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_pred_loss += pred_loss.item()

        # end of epoch eval
        print(f"End of epoch {epoch}")

        # Save epoch losses
        train_recon_losses.append(train_recon_loss / batches_per_epoch)
        train_kl_losses.append(train_kl_loss / batches_per_epoch)
        train_pred_losses.append(train_pred_loss / batches_per_epoch)

        plot_losses(train_recon_losses, train_kl_losses, train_pred_losses, filename="train")

        vae.eval()

        recon_test, _, _, _ = vae.forward(test_sample)
        recon_train, _, _, _ = vae.forward(train_sample)

        expected_test = hot_to_smiles(test_sample.detach().cpu().numpy(), indices_char)[0]
        computed_test = hot_to_smiles(recon_test.detach().cpu().numpy(), indices_char)[0]
        
        expected_train = hot_to_smiles(train_sample.detach().cpu().numpy(), indices_char)[0]
        computed_train = hot_to_smiles(recon_train.detach().cpu().numpy(), indices_char)[0]


        print("TRAIN: ")
        # Print train strings (target and output)
        print(f"Target: {expected_train:<{max_length}}")
        print(f"Output: {computed_train:<{max_length}}")

        print("TEST: ")
        # Print test strings (target and output)
        print(f"Target: {expected_test:<{max_length}}")
        print(f"Output: {computed_test:<{max_length}}")

    
    # Save model weights
    torch.save(decoder.state_dict(), str(params["exp_path"] / params['decoder_weights_file']))
    torch.save(encoder.state_dict(), str(params["exp_path"] / params['encoder_weights_file']))
    torch.save(property_predictor.state_dict(), str(params["exp_path"] / params['prop_pred_weights_file']))
    

    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp',
                        help='experiment name', default='test')
    args = vars(parser.parse_args())

    # Get the main directory absolute path
    main_dir = Path(__file__).resolve().parent.parent
    exp_path = main_dir / os.path.join("exps", args["exp"])
    if not exp_path.exists():
        # Create the folder
        exp_path.mkdir(parents=True)
    
    params = hyperparameters.load_params(exp_path / "exp.json")
    params["char_file"] = main_dir / os.path.join("data", params["char_file"])
    params["data_file"] = main_dir / os.path.join("data", params["data_file"])

    if params["pretrained"]:

        pretrained_weights_path = main_dir / "exps" / params["pretrained_weights"]
        if not pretrained_weights_path.exists():
            raise NameError(f"Pretrained weights folder {pretrained_weights_path} does not exist.")

        params["pretrained_encoder_file"] = pretrained_weights_path / "encoder.pt"
        # params["pretrained_decoder_file"] = pretrained_weights_path / "decoder.pt"
        params["pretrained_predictor_file"] = pretrained_weights_path / "prop_pred.pt"

    params["exp_path"] = exp_path
    
    if "test_idx_file" in params.keys():
        params['test_idx_file'] = params["exp_path"] / params['test_idx_file']

    print("All params:", params)
    print("GPU available? {}".format(torch.cuda.is_available()))

    train(params)