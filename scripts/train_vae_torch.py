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
import yaml
import chemvae_torch.mol_utils as mu

# Define the models, loss functions, etc.
# You need to define your PyTorch models and other necessary components here.
# This includes defining the encoder, decoder, property predictor, loss functions,
# and any other required components.

# For demonstration purposes, let's assume you have defined your PyTorch models as follows:

# encoder = YourEncoderModel()
# decoder = YourDecoderModel()
# property_predictor = YourPropertyPredictorModel()

def hot_to_smiles(hot_x, indices_chars):
    
    assert type(hot_x) == np.ndarray
    assert len(hot_x.shape) == 3
    
    smiles = []
    for i in range(hot_x.shape[0]):  # number of samples
        temp_str = ""
        for j in range(hot_x.shape[1]):  # length of smiles
            index = np.argmax(hot_x[i, j, :])
            temp_str += indices_chars[index]
        smiles.append(temp_str)
    return smiles

def smiles_to_hot(smiles, params, canonize_smiles=True, check_smiles=False):

    chars = yaml.safe_load(open(params["char_file"]))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    if isinstance(smiles, str):
        smiles = [smiles]

    if canonize_smiles:
        smiles = [mu.canon_smiles(s) for s in smiles]

    if check_smiles:
        smiles = mu.smiles_to_hot_filter(smiles, char_indices)

    z = mu.smiles_to_hot(smiles, params["MAX_LEN"], params["PADDING"], char_indices, params["NCHARS"])
    return z

def vectorize_data(params):
    # @out : Y_train /Y_test : each is list of datasets.
    #        i.e. if reg_tasks only : Y_train_reg = Y_train[0]
    #             if logit_tasks only : Y_train_logit = Y_train[0]
    #             if both reg and logit_tasks : Y_train_reg = Y_train[0], Y_train_reg = 1
    #             if no prop tasks : Y_train = []

    MAX_LEN = params['MAX_LEN']

    CHARS = yaml.safe_load(open(params['char_file']))
    params['NCHARS'] = len(CHARS)
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))
    #INDICES_CHAR = dict((i, c) for i, c in enumerate(CHARS))

    ## Load data for properties
    if params['do_prop_pred'] and ('data_file' in params):
        if "data_normalization_out" in params:
            normalize_out = params["data_normalization_out"]
        else:
            normalize_out = None

        ################
        if ("reg_prop_tasks" in params) and ("logit_prop_tasks" in params):
            smiles, Y_reg, Y_logit = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN,
                    reg_tasks=params['reg_prop_tasks'], logit_tasks=params['logit_prop_tasks'],
                    normalize_out = normalize_out)
        elif "logit_prop_tasks" in params:
            smiles, Y_logit = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN,
                    logit_tasks=params['logit_prop_tasks'], normalize_out=normalize_out)
        elif "reg_prop_tasks" in params:
            smiles, Y_reg = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN,
                    reg_tasks=params['reg_prop_tasks'], normalize_out=normalize_out)
        else:
            raise ValueError("please sepcify logit and/or reg tasks")

    ## Load data if no properties
    else:
        smiles = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN)

    # print("NUMBER OF SAMPLES L<120: ", len(smiles))

    if 'limit_data' in params.keys():
        sample_idx = np.random.choice(np.arange(len(smiles)), params['limit_data'], replace=False)
        smiles=list(np.array(smiles)[sample_idx])
        if params['do_prop_pred'] and ('data_file' in params):
            if "reg_prop_tasks" in params:
                Y_reg =  Y_reg[sample_idx]
            if "logit_prop_tasks" in params:
                Y_logit =  Y_logit[sample_idx]

    print('Training set size is', len(smiles))
    print('first smiles: \"', smiles[0], '\"')
    print('total chars:', NCHARS)

    print('Vectorization...')
    X = mu.smiles_to_hot(smiles, MAX_LEN, params[
                             'PADDING'], CHAR_INDICES, NCHARS)

    print('Total Data size', X.shape[0])
    if np.shape(X)[0] % params['batch_size'] != 0:
        X = X[:np.shape(X)[0] // params['batch_size']
              * params['batch_size']]
        if params['do_prop_pred']:
            if "reg_prop_tasks" in params:
                Y_reg = Y_reg[:np.shape(Y_reg)[0] // params['batch_size']
                      * params['batch_size']]
            if "logit_prop_tasks" in params:
                Y_logit = Y_logit[:np.shape(Y_logit)[0] // params['batch_size']
                      * params['batch_size']]

    np.random.seed(params['RAND_SEED'])
    rand_idx = np.arange(np.shape(X)[0])
    np.random.shuffle(rand_idx)

    TRAIN_FRAC = 1 - params['val_split']
    num_train = int(X.shape[0] * TRAIN_FRAC)

    if num_train % params['batch_size'] != 0:
        num_train = num_train // params['batch_size'] * \
            params['batch_size']

    train_idx, test_idx = rand_idx[: int(num_train)], rand_idx[int(num_train):]

    if 'test_idx_file' in params.keys():
        np.save(params['test_idx_file'], test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    print('shape of input vector : {}', np.shape(X_train))
    print('Training set size is {}, after filtering to max length of {}'.format(
        np.shape(X_train), MAX_LEN))

    if params['do_prop_pred']:
        # !# add Y_train and Y_test here
        Y_train = []
        Y_test = []
        if "reg_prop_tasks" in params:
            Y_reg_train, Y_reg_test = Y_reg[train_idx], Y_reg[test_idx]
            Y_train.append(Y_reg_train)
            Y_test.append(Y_reg_test)
        if "logit_prop_tasks" in params:
            Y_logit_train, Y_logit_test = Y_logit[train_idx], Y_logit[test_idx]
            Y_train.append(Y_logit_train)
            Y_test.append(Y_logit_test)

        return X_train, X_test, Y_train, Y_test

    else:
        return X_train, X_test


def train_decoder(params):
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
    checkpoints_path = Path(__file__).resolve().parent.parent / "checkpoints/zinc_properties" 
    if not params["train_all"]:
        encoder.load_state_dict(torch.load(str(checkpoints_path / params['pretrained_encoder_file'])))
        decoder.load_state_dict(torch.load(str(checkpoints_path / params['pretrained_decoder_file'])))
        property_predictor.load_state_dict(torch.load(str(checkpoints_path / params['pretrained_predictor_file'])))

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
    
    vae = AE_PP_Model(encoder, decoder, property_predictor, params)

    # move models to GPU
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    property_predictor = property_predictor.to(device)
    vae = vae.to(device)

    if not params["train_all"]:
        # Freeze encoder
        for param in vae.encoder.parameters():
            param.requires_grad = False
        
        # Freeze property predictor
        for param in vae.property_predictor.parameters():
            param.requires_grad = False

    # Define optimizer
    if params['optim'] == 'adam':
        optimizer = optim.Adam(vae.parameters(), lr=params['lr'], betas=(params['momentum'], 0.999))
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

    # retrieving character encoding
    chars = yaml.safe_load(open(params["char_file"]))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # Train the model
    for epoch in range(params['prev_epochs'], params['epochs']):
        
        for batch_idx, (x, y) in enumerate(train_loader):

            vae.train()
            optimizer.zero_grad()

            # update KL loss weight based on schedule
            kl_loss_weight = sigmoid_schedule(
                epoch,
                slope=params["anneal_sigmod_slope"],
                start=params["vae_annealer_start"],
                weight_orig=params["kl_loss_weight"]
            )

            # Forward pass
            reconstruction, prediction, mu, logvar = vae.forward(x)
            loss, recon_loss, kl_loss, pred_loss = vae.loss_function(
                reconstruction=reconstruction, 
                prediction=prediction, 
                mu=mu, 
                logvar=logvar, 
                x=x, 
                y=y,
                kl_loss_weight=kl_loss_weight,
            )

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # eval
            vae.eval()
            recon_test, _, _, _ = vae.forward(test_sample)

            expected = hot_to_smiles(test_sample.detach().cpu().numpy(), indices_char)[0]
            computed = hot_to_smiles(recon_test.detach().cpu().numpy(), indices_char)[0]

            max_length = max(len(expected), len(computed))

            # Format and print the strings with alignment
            print("Epoch {} Batch {}.".format(epoch, batch_idx))

            print(f"Losses - REC: {recon_loss.item()} \
                  KL: {kl_loss.item()} \
                  PRED: {pred_loss.item()} \
                  TOT: {loss.item()}")
            
            # Print test strings (target and output)
            print(f"Target: {expected:<{max_length}}")
            print(f"Output: {computed:<{max_length}}")

        # Print some metrics or do logging
        # if epoch % params['log_interval'] == 0:
        #     print(f"Epoch {epoch}: Recon Loss: {recon_loss.item()}. Total Loss: {loss.item()}")
    
    # Save model weights
    exp_path = Path(__file__).resolve().parent.parent / "exps" / params["name"]
    if not exp_path.exists():
        # Create the folder
        exp_path.mkdir(parents=True)

    torch.save(decoder.state_dict(), str(exp_path / params['decoder_torch_weights_file']))
    torch.save(encoder.state_dict(), str(exp_path / params['encoder_torch_weights_file']))
    torch.save(property_predictor.state_dict(), str(exp_path / params['prop_pred_torch_weights_file']))
    

    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')

    return

def sigmoid_schedule(time_step, slope=1.0, start=None, weight_orig=None):
    """
    Sigmoid annealing.

    :param time_step: epoch number
    :param slope: slope of the sigmoid
    :param start: starting point of the annealing
    :param weight_orig: value of the weight at time_step=0
    :return: sigmoid annealing weight
    """
    # Inverted float(time_step) and start wrt the original function
    # The function should be decreasing with the time_step for weight annealing
    return weight_orig * float(1 / (1.0 + np.exp(slope * (float(time_step) - start))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_file',
                        help='experiment file', default='exp.json')
    args = vars(parser.parse_args())

    # Get the main directory absolute path
    main_dir = Path(__file__).resolve().parent.parent
    relative_path = os.path.join("checkpoints/zinc_properties", args["exp_file"])
    args['exp_file'] = main_dir / relative_path
    
    params = hyperparameters.load_params(args['exp_file'])
    params["char_file"] = main_dir / os.path.join("checkpoints/zinc_properties", params["char_file"])
    params["data_file"] = main_dir / os.path.join("checkpoints/zinc_properties", params["data_file"])

    params["pretrained_encoder_file"] = main_dir / "checkpoints/zinc_properties/pretrained_encoder.pt"
    params["pretrained_decoder_file"] = main_dir / "checkpoints/zinc_properties/pretrained_decoder.pt"
    params["pretrained_predictor_file"] = main_dir / "checkpoints/zinc_properties/pretrained_predictor.pt"

    print("All params:", params)

    print("GPU available? {}".format(torch.cuda.is_available()))

    train_decoder(params)