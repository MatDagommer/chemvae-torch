import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from functools import partial
from models_torch import EncoderModel, DecoderModel, PropertyPredictorModel, AE_PP_Model
import hyperparameters
import argparse
from pathlib import Path
import os
import yaml
import mol_utils as mu

# Define the models, loss functions, etc.
# You need to define your PyTorch models and other necessary components here.
# This includes defining the encoder, decoder, property predictor, loss functions,
# and any other required components.

# For demonstration purposes, let's assume you have defined your PyTorch models as follows:

# encoder = YourEncoderModel()
# decoder = YourDecoderModel()
# property_predictor = YourPropertyPredictorModel()

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

    # Load data
    X_train, X_test, Y_train, Y_test = vectorize_data(params)

    encoder = EncoderModel(params)
    decoder = DecoderModel(params)
    property_predictor = PropertyPredictorModel(params)

    # loss weights
    #NOTE: In original implementation, xent_loss_weight and kl_loss_var are trainable
    xent_loss_weight = params['xent_loss_weight']
    ae_loss_weight = 1. - params['prop_pred_loss_weight']
    kl_loss_var = params['kl_loss_weight']
    prop_pred_loss_weight = params['prop_pred_loss_weight']
    
    model_loss_weights = {
                    'reconstruction_loss': ae_loss_weight * xent_loss_weight,
                    'kl_loss': ae_loss_weight * kl_loss_var,
                    'prediction_loss': prop_pred_loss_weight
                    }
    
    params["model_loss_weights"] = model_loss_weights
    
    vae = AE_PP_Model(encoder, decoder, property_predictor, params)

    # Load pretrained parameters
    # TODO: Write this part

    # Freeze encoder
    for param in vae.encoder.parameters():
        param.requires_grad = False
    # Freeze property predictor
    for param in vae.property_predictor.parameters():
        param.requires_grad = False

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
        
        property_predictor.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            reconstruction, prediction, mu, logvar = vae.forward(x)
            loss = vae.loss_function(reconstruction, prediction, mu, logvar, x, y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Print some metrics or do logging
        if epoch % params['log_interval'] == 0:
            print(f"Epoch {epoch}: Loss: {loss.item()}")

    # Save the new decoder
    torch.save(decoder.state_dict(), params['decoder_torch_weights_file'])

    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')

    return


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
    
    print("All params:", params)
    train_decoder(params)