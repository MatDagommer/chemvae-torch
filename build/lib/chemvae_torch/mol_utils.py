"""Functions for processing SMILES strings and dataframes of SMILES strings."""

import logging
import pickle as pkl

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as Chem

logging.getLogger("autoencoder")
logging.getLogger().setLevel(20)
logging.getLogger().addHandler(logging.StreamHandler())


# =================
# text io functions
# ==================


def smiles_to_mol(smiles):
    """
    Convert a SMILES string to an RDKit molecule object.

    :param smiles: SMILES string
    :return: RDKit molecule object
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception as e:
        print("ERROR: Check smiles file. Bad SMILES:", e)
        pass
    return None


def verify_smiles(smile):
    """
    Check if a SMILES string is valid.

    :param smile: SMILES string
    :return: True if valid, False otherwise
    """
    return (smile != "") and pd.notnull(smile) and (Chem.MolFromSmiles(smile) is not None)


def good_smiles(smile):
    """
    Check if a SMILES string is valid and has no stereochemistry.

    :param smile: SMILES string
    :return: True if valid, False otherwise
    """
    if verify_smiles(smile):
        return canon_smiles(smile)
    else:
        return None


def pad_smile(string, max_len, padding="right"):
    """
    Pad a SMILES string to a specified length.

    :param string: SMILES string
    :param max_len: maximum length of SMILES string
    :param padding: "right" or "left"
    :return: padded SMILES string
    """
    if len(string) <= max_len:
        if padding == "right":
            return string + " " * (max_len - len(string))
        elif padding == "left":
            return " " * (max_len - len(string)) + string
        elif padding == "none":
            return string


def filter_valid_length(strings, max_len):
    """
    Filter a list of SMILES strings by length.

    :param strings: list of SMILES strings
    :param max_len: maximum length of SMILES string
    :return: list of SMILES strings with length <= max_len
    """
    return [s for s in strings if len(s) <= max_len]


def filter_valid_smiles_return_invalid(strings, max_len):
    """
    Filter a list of SMILES strings by length.

    :param strings: list of SMILES strings
    :param max_len: maximum length of SMILES string
    :return: list of SMILES strings with length <= max_len
    """
    filter_list = []
    new_smiles = []
    for idx, s in enumerate(strings):
        if len(s) > max_len:
            filter_list.append(idx)
        else:
            new_smiles.append(s)
    return new_smiles, filter_list


def smiles_to_hot(smiles, max_len, padding, char_indices, nchars):
    """
    Convert a list of SMILES strings to a one-hot representation.

    :param smiles: list of SMILES strings
    :param max_len: maximum length of SMILES string
    :param padding: "right" or "left"
    :param char_indices: dictionary mapping characters to indices
    :param nchars: number of characters in the character set
    :return: one-hot representation of SMILES strings
    :raises KeyError: if a character is not in the character set
    """
    smiles = [pad_smile(i, max_len, padding) for i in smiles if pad_smile(i, max_len, padding)]

    X = np.zeros((len(smiles), max_len, nchars), dtype=np.float32)

    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                X[i, t, char_indices[char]] = 1
            except KeyError as e:
                print("ERROR: Check chars file. Bad SMILES:", smile)
                raise e
    return X


def smiles_to_hot_filter(smiles, char_indices):
    """
    Filter a list of SMILES strings by length.

    :param smiles: list of SMILES strings
    :param char_indices: dictionary mapping characters to indices
    :return: list of SMILES strings with length <= max_len
    """
    filtered_smiles = []
    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                char_indices[char]
            except KeyError:
                break
        else:
            filtered_smiles.append(smile)
    return filtered_smiles


def term_hot_to_smiles(x, temperature, indices_chars):
    """
    Convert a one-hot representation to a SMILES string.

    :param x: one-hot representation of a SMILES string
    :param temperature: amount of noise for sampling the final output
    :param indices_chars: dictionary mapping indices to characters
    :return: SMILES string
    """
    temp_string = ""
    for j in x:
        index = thermal_argmax(j, temperature)
        temp_string += indices_chars[index]
    return temp_string


def hot_to_smiles(hot_x, indices_chars):
    """
    Convert a one-hot representation to a SMILES string.

    :param hot_x: one-hot representation of a SMILES string
    :param indices_chars: dictionary mapping indices to characters
    :return: SMILES string
    """
    smiles = []
    for x in hot_x:
        temp_str = ""
        for j in x:
            index = np.argmax(j)
            temp_str += indices_chars[index]
        smiles.append(temp_str)
    return smiles


def thermal_argmax(prob_arr, temperature):
    """
    Sample from a multinomial distribution.

    :param prob_arr: array of probabilities
    :param temperature: amount of noise for sampling the final output
    :return: index of the sampled element
    """
    prob_arr = np.log(prob_arr) / temperature
    prob_arr = np.exp(prob_arr) / np.sum(np.exp(prob_arr))
    print(prob_arr)
    if np.greater_equal(prob_arr.sum(), 1.0000000001):
        logging.warn("Probabilities to sample add to more than 1, {}".format(prob_arr.sum()))
        prob_arr = prob_arr / (prob_arr.sum() + 0.0000000001)
    if np.greater_equal(prob_arr.sum(), 1.0000000001):
        logging.warn("Probabilities to sample still add to more than 1")
    return np.argmax(np.random.multinomial(1, prob_arr, 1))


def load_smiles(smi_file, max_len=None, return_filtered=False):
    """
    Load a list of SMILES strings from a file.

    :param smi_file: path to file containing SMILES strings
    :param max_len: maximum length of SMILES string
    :param return_filtered: whether to return filtered SMILES strings
    :return: list of SMILES strings
    """
    if smi_file[-4:] == ".pkl":
        with open(smi_file, "rb") as f:
            smiles = pkl.load(f)
    else:  # assume file is a text file
        with open(smi_file, "r") as f:
            smiles = f.readlines()
        smiles = [i.strip() for i in smiles]

    if max_len is not None:
        if return_filtered:
            smiles, filtrate = filter_valid_smiles_return_invalid(smiles, max_len)
            if len(filtrate) > 0:
                print("Filtered {} smiles due to length".format(len(filtrate)))
            return smiles, filtrate

        else:
            old_len = len(smiles)
            smiles = filter_valid_length(smiles, max_len)
            diff_len = old_len - len(smiles)
            if diff_len != 0:
                print("Filtered {} smiles due to length".format(diff_len))

    return smiles


def load_smiles_and_data_df(data_file, max_len, reg_tasks=None, logit_tasks=None, normalize_out=None, dtype="float64"):
    """
    Load SMILES strings and data from a CSV file.

    :param data_file: path to CSV file containing SMILES strings and data
    :param max_len: maximum length of SMILES string
    :param reg_tasks: list of columns in df that correspond to regression tasks
    :param logit_tasks: list of columns in df that correspond to logit tasks
    :param normalize_out: path to CSV file to write normalization parameters
    :param dtype: data type
    :return: list of SMILES strings, regression data, and logit data
    """
    # reg_tasks : list of columns in df that correspond to regression tasks
    # logit_tasks : list of columns in df that correspond to logit tasks

    if logit_tasks is None:
        logit_tasks = []
    if reg_tasks is None:
        reg_tasks = []
    df = pd.read_csv(data_file)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip()
    df = df[df.iloc[:, 0].str.len() <= max_len]
    smiles = df.iloc[:, 0].tolist()

    reg_data_df = df[reg_tasks]
    logit_data_df = df[logit_tasks]
    # Load regression tasks
    if len(reg_tasks) != 0 and normalize_out is not None:
        df_norm = pd.DataFrame(reg_data_df.mean(axis=0), columns=["mean"])
        df_norm["std"] = reg_data_df.std(axis=0)
        reg_data_df = (reg_data_df - df_norm["mean"]) / df_norm["std"]
        df_norm.to_csv(normalize_out)

    if len(logit_tasks) != 0 and len(reg_tasks) != 0:
        return smiles, np.vstack(reg_data_df.values).astype(dtype), np.vstack(logit_data_df.values).astype(dtype)
    elif len(reg_tasks) != 0:
        return smiles, np.vstack(reg_data_df.values).astype(dtype)
    elif len(logit_tasks) != 0:
        return smiles, np.vstack(logit_data_df.values).astype(dtype)
    else:
        return smiles


def smiles2one_hot_chars(smi_list, max_len):
    """
    Convert a list of SMILES strings to a one-hot representation.

    :param smi_list: list of SMILES strings
    :param max_len: maximum length of SMILES string
    :return: one-hot representation of SMILES strings
    """
    # get all the characters
    char_lists = [list(smi) for smi in smi_list]
    chars = list(set([char for sub_list in char_lists for char in sub_list]))
    chars.append(" ")

    return chars


def make_charset(smi_file, char_file):
    """
    Create a character set from a list of SMILES strings.

    :param smi_file: path to file containing SMILES strings
    :param char_file: path to file to write character set
    :return: list of unique characters
    """
    with open(smi_file, "r") as afile:
        unique_chars = set(afile.read())
    bad = ["\n", '"']
    unique_chars = [c for c in unique_chars if c not in bad]
    unique_chars.append(" ")
    print("found {} unique chars".format(len(unique_chars)))
    astr = str(unique_chars).replace("'", '"')
    print(astr)

    with open(char_file, "w") as afile:
        afile.write(astr)

    return


# =================
# data parsing io functions
# ==================


def CheckSmiFeasible(smi):
    """
    Check if a SMILES string is valid.

    :param smi: SMILES string
    :return: True if valid, False otherwise
    """
    # See if you can make a smiles with mol object
    #    if you can't, then skip
    try:
        get_molecule_smi(Chem.MolFromSmiles(smi))
    except Exception as e:
        print("ERROR: Check smiles file. Bad SMILES:", e)
        return False
    return True


def balanced_parentheses(input_string):
    """
    Check if a SMILES string has balanced parentheses.

    :param input_string: SMILES string
    :return: True if balanced, False otherwise
    """
    s = []
    balanced = True
    index = 0
    while index < len(input_string) and balanced:
        token = input_string[index]
        if token == "(":
            s.append(token)
        elif token == ")":
            if len(s) == 0:
                balanced = False
            else:
                s.pop()

        index += 1

    return balanced and len(s) == 0


def matched_ring(s):
    """
    Check if a SMILES string has matched rings.

    :param s: SMILES string
    :return: True if matched, False otherwise
    """
    return s.count("1") % 2 == 0 and s.count("2") % 2 == 0


def fast_verify(s):
    """
    Check if a SMILES string is valid.

    :param s: SMILES string
    :return: True if valid, False otherwise
    """
    return matched_ring(s) and balanced_parentheses(s)


def get_molecule_smi(mol_obj):
    """
    Convert an RDKit molecule object to a SMILES string.

    :param mol_obj: RDKit molecule object
    :return: SMILES string
    """
    return Chem.MolToSmiles(mol_obj)


def canon_smiles(smi):
    """
    Convert a SMILES string to a canonical SMILES string.

    :param smi: SMILES string
    :return: canonical SMILES string
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)


if __name__ == "__main__":
    # print("please import me")
    smiles, reg_dat, logit_dat = load_smiles_and_data_df(
        "zinc/250k_rndm_zinc_drugs_clean_5.csv", 120, ["logP", "qed", "SAS"], ["NRingsGT6", "PAINS"]
    )
    print(smiles[:5])
    print(reg_dat[:5, :])
    print(logit_dat[:5, :])
