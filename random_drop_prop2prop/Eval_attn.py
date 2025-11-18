"""
Initialized from prop2prop/Train_attn.py
(commit: 8f23cd9ea3b630d1903fbe0a559a09093327c4a7)

Modified preprocess_data():
  - add the generation of smiles training/validation data
  - modify outputs

Modified setup_data_loaders():
  - add calls to DynamicSMILESDataset class
    - combination of DynamicDropDataset and SMILESDataset
  - modify inputs
  - To-Do: Verify this is correct!!!

Do not read in 'properties_removed' config file input
when initializing the model.
"""

"""
This Evaluation Script is not a simple modification
of the training script like the other regimes' Eval
scripts are.
To properly save the prediction data, the appropriate
fraction was dropped initially, and then not dynamically.
As such, random dropping was removed from the dataset.

Do not use the modules here for something else.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,  Dataset
from tqdm import tqdm
import argparse
import json
import importlib
from torchsummary import summary
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from collections.abc import Iterable
from transformers import AutoTokenizer, AutoModel


def main(config):

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    model_epochs = config['model_epochs']
    checkpoint_fp = os.path.join(
        config['working_directory'], config['model_save_prefix']) + f'_epoch_{model_epochs-1}.pt'
    print(f"Model checkpoints will be written to {checkpoint_fp}")

    if config['load_ex_model'] and config['model_epochs'] == 0:
        print('Caution: You have elected to load an existing model to train (load_ex_model == True) but have set model_epochs to zero.  Double check before proceeding.')
    if not config['load_ex_model'] and config['model_epochs'] != 0:
        sys.exit("Error: If you have a nonzero model_steps argument (meaning you want to continue training an existing model), you also need to change load_ex_model to True.")

    # Load and preprocess data
    print("Preprocessing data")
    train_x, train_y, val_x, val_y = preprocess_data(
        config
    )
    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(val_y.shape)
    
    # Import Model Class
    model_module = importlib.import_module(config['model_file'])
    model_class = getattr(model_module, config['model_class'])

    # Instantiate the model and load existing model if applicable
    quad_on = config.get('include_quad', True)
    model = model_class(include_quad=quad_on)

    # TO-DO: Make a separate JSON entry to control where to look for pre-existing model
    checkpoint_fp = os.path.join(
        config['working_directory'], config['model_save_prefix']
    ) + f'_cp_overall.pt'
    model.load_state_dict(torch.load(checkpoint_fp))

    # Load Optimizer and Loss Function
    learning_rate = config.get('max_lr', 1e-06)
    n_warmup = config.get('warmup_steps', 4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if config['var_lr']:
        sched = ScheduledOptim(optimizer, lr_mul=learning_rate, d_model=64,
                               n_warmup_steps=n_warmup, n_current_steps=config['model_epochs']
                               )
    else:
        sched = optimizer

    loss_function = load_loss_function(config)
    loss_tracker = PropertyMSE()

    for j in range(1,10):
        drop_frac = j / 10

        val_prop, mask = perm_mask(drop_frac, val_x)



        # Setup DataLoader
        print("Initializing train/validate data loaders")
        batch_size = config.get('batch_size', 500)
        drop_frac = config.get('fraction_dropped', 0.0)
        val_loader = setup_data_loaders(
            train_x, train_y,
            val_prop, val_y,
            batch_size, drop_frac
        )
        print(len(val_loader))
        train_loader = []
        # Execute Training and Validation
        predictions = train_and_validate(
            config,
            model,
            sched,
            loss_function,
            loss_tracker,
            train_loader,
            val_loader,
            device='cuda',
        )

        for i in range(val_prop.shape[1]):  # Loop over the 22 properties
            # Filter the rows where mask[:, i] == 1 for the current property
            mask_i = mask[:, i]
            filtered_targets = val_y[mask_i == 1, i]
            filtered_predictions = predictions[mask_i == 1, i]
    
            # Create a DataFrame with the filtered data
            df = pd.DataFrame({
                'target': filtered_targets,
                'prediction': filtered_predictions
            })
    
            # Save to a CSV file
            filename = os.path.join(config['working_directory'], config['model_save_prefix']) + f'_eval_{j}_prop_{i}_test.csv'
            df.to_csv(filename, index=False)
            print(f"Saved {filename} with {len(df)} rows.")

def perm_mask(frac_dropped, props):
    shape = props.shape
    np.random.seed(42)  # Set seed for reproducibility

    # Generate the binary mask
    binary_mask = (np.random.rand(*shape) < frac_dropped).astype(int) # 1 is masked, 0 if left alone
    prop_masked = -100 * binary_mask + props * (1-binary_mask)
    return prop_masked.astype(float), binary_mask

def preprocess_data(config):

    train_comb = np.loadtxt(os.path.join(config['working_directory'], config['data_path_train']), delimiter=',')
    val_comb = np.loadtxt(os.path.join(config['working_directory'], config['data_path_val']), delimiter=',')
    print(train_comb.shape)
    print(val_comb.shape)
    condition_train = (np.abs(train_comb[:, 18]) <= 0.2) & (np.abs(train_comb[:, 20]) <= 0.2)
    train_comb = train_comb[condition_train]

    condition_val = (np.abs(val_comb[:, 18]) <= 0.2) & (np.abs(val_comb[:, 20]) <= 0.2)
    val_comb = val_comb[condition_val]

    if config['rem_outliers']:
        # Load property bounds
        with open(os.path.join(config['working_directory'], config['outlier_bound_fp']), 'r') as f:
            bounds = json.load(f)

        # Remove outliers based on bounds
        for prop, bound in bounds.items():
            prop_idx = int(prop)
            min_val, max_val = bound['min'], bound['max']
            train_comb = train_comb[(train_comb[:, prop_idx] >= min_val) & (train_comb[:, prop_idx] <= max_val)]
            val_comb = val_comb[(val_comb[:, prop_idx] >= min_val) & (val_comb[:, prop_idx] <= max_val)]

    print(f"Shape after removing outliers: Train = {train_comb.shape}, Validation = {val_comb.shape}")

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))

    # Fit the scaler to the data and transform it
    train_comb = scaler.fit_transform(train_comb)
    val_comb = scaler.transform(val_comb)

    # Make sure no validation values interfere with -100 missing token.
    assert np.all(val_comb > -99), "Error: Validation data contains values less than -99! Will interfere with missing tokens!"

    # Warning for values less than -10
    rows_with_ood = np.any(val_comb < -10, axis=1)  # Boolean array indicating rows with values < -10
    num_ood_rows = np.sum(rows_with_ood)  # Count the number of such rows
    if num_ood_rows > 0:
        print(f"WARNING! You have {num_ood_rows} validation data points that are out of distribution.")

    if config['include_quad']:
        train_x = train_comb.copy()
        val_x = val_comb.copy()
    else:
        # Include only the first 22 properties
        train_x = train_comb[:, :22].copy()
        val_x = val_comb[:, :22].copy()

    train_y = train_comb[:, :22].copy()
    val_y = val_comb[:, :22].copy()

    return train_x, train_x, val_x, val_x

'''
def preprocess_data(config):

    # Load the CSV file into a DataFrame
    data = pd.read_csv(config['data_path'])

    # Extract the SMILES and property data for the 'train' split
    train_data = data[data['split'] == 'train']
    train_smiles = train_data['smiles'].to_numpy()
    train_properties = train_data.drop(columns=['smiles', 'split']).to_numpy()

    # Extract the SMILES and property data for the 'val' split
    val_data = data[data['split'] == 'val']
    val_smiles = val_data['smiles'].to_numpy()
    val_properties = val_data.drop(columns=['smiles', 'split']).to_numpy()
    print(val_properties.shape)  # debug
    print(len(val_smiles))  # debug

    print(train_properties.shape)  # debug
    print(val_properties.shape)  # debug

    # Remove really crazy outliers from props 18 and 20
    condition_train = (np.abs(train_properties[:, 18]) <= 0.2) & (
        np.abs(train_properties[:, 20]) <= 0.2)
    train_properties = train_properties[condition_train]
    train_smiles = train_smiles[condition_train]
    print(train_properties.shape)  # debug
    print(len(train_smiles))  # debug

    condition_val = (np.abs(val_properties[:, 18]) <= 0.2) & (
        np.abs(val_properties[:, 20]) <= 0.2)
    val_properties = val_properties[condition_val]
    val_smiles = val_smiles[condition_val]
    print(val_properties.shape)  # debug
    print(len(val_smiles))  # debug

    # Removing outliers from the rest of the properties
    if config['rem_outliers']:
        # Load property bounds
        with open(config['outlier_bound_fp'], 'r') as f:
            bounds = json.load(f)

        # Remove outliers based on bounds
        for prop, bound in bounds.items():
            prop_idx = int(prop)
            min_val, max_val = bound['min'], bound['max']
            train_test = (train_properties[:, prop_idx] >= min_val) & (
                train_properties[:, prop_idx] <= max_val)
            train_properties = train_properties[train_test]
            train_smiles = train_smiles[train_test]
            val_test = (val_properties[:, prop_idx] >= min_val) & (
                val_properties[:, prop_idx] <= max_val)
            val_properties = val_properties[val_test]
            val_smiles = val_smiles[val_test]

    # debug
    print(
        f"Shape after removing outliers: Train = {train_properties.shape}, Validation = {val_properties.shape}")
    print(
        f"Shape after removing outliers: Train = {train_smiles.shape}, Validation = {val_smiles.shape}")

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))

    # Fit the scaler to the data and transform it
    train_properties = scaler.fit_transform(train_properties)
    val_properties = scaler.transform(val_properties)

    # Make sure no validation values interfere with -100 missing token.
    assert np.all(val_properties > -
                  99), "Error: Validation data contains values less than -99! Will interfere with missing tokens!"

    # Warning for validation values less than -10
    # Boolean array indicating rows with values < -10
    rows_with_ood = np.any(val_properties < -10, axis=1)
    num_ood_rows = np.sum(rows_with_ood)  # Count the number of such rows
    if num_ood_rows > 0:
        print(
            f"WARNING! You have {num_ood_rows} validation data points that are out of distribution.")

    if config['include_quad']:
        train_x = train_properties.copy()
        val_x = val_properties.copy()
    else:
        # Include only the first 22 properties
        train_x = train_properties[:, :22].copy()
        val_x = val_properties[:, :22].copy()

    print(val_smiles.shape)
    print(f'Max len train: {max([len(smi) for smi in train_smiles])}')
    print(f'Max len val: {max([len(smi) for smi in val_smiles])}')

    return train_x, train_smiles, train_x, val_x, val_smiles, val_x
'''

class PropertyMSE(nn.Module):
    """
    Module to calculate the Mean Squared Error (MSE) for each property along dim=1.

    This can be used for tracking loss per property or integrated into a training pipeline.
    """

    def __init__(self):
        super(PropertyMSE, self).__init__()

    def forward(self, predictions, targets):
        """
        Forward pass to compute the MSE for each property.

        Args:
            predictions (torch.Tensor): The model's output of shape (batch_size, num_properties).
            targets (torch.Tensor): The ground truth of shape (batch_size, num_properties).

        Returns:
            torch.Tensor: An iterable tensor of MSE values for each property.
        """
        # Ensure the predictions and targets have the same shape
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} and targets {targets.shape}")

        # Compute MSE along dim=0 (property-wise across the batch)
        mse_per_property = torch.mean((predictions - targets) ** 2, dim=0)
        return mse_per_property


def setup_data_loaders(train_x, train_y, val_x, val_y, batch_size, drop_frac):
    val_x = torch.tensor(val_x, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)
    val_dataset = DynamicSMILESDataset(
        val_x, val_y, drop_probability=drop_frac)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return val_loader

# Which dataset to use? SMILES or DynamicDrop? Or a Frankenstein's monster of both?


class DynamicSMILESDataset(Dataset):
    """
    Combine property data and smiles data into one cool data set

    Apply random dropping masking to property data (from user input --> config)
    """

    def __init__(self, inputs, targets, drop_probability=0.0,
                 model_name="DeepChem/ChemBERTa-77M-MLM", max_length=128,
                 missing_token=-100
                 ):
        self.inputs = inputs                # Corresponding input data
        self.targets = targets              # Corresponding target data
        self.max_length = max_length
        self.drop_probability = drop_probability
        self.missing_token = missing_token

    def __len__(self):
        return len(self.inputs)  # the first dimmension will be the smiles

    def __getitem__(self, idx):
        # Retrieve the input and target
        # copy inputs so masking doesn't carry forward from epoch to epoch
        inputs = self.inputs[idx].clone()
        targets = self.targets[idx]

        return inputs, targets

# class SMILESDataset(Dataset):
#     def __init__(self, inputs, smiles_list, targets, model_name="DeepChem/ChemBERTa-77M-MLM", max_length=128):
#         self.smiles_list = smiles_list      # List of SMILES strings
#         self.inputs = inputs                # Corresponding input data
#         self.targets = targets              # Corresponding target data
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.smiles_list)

#     def __getitem__(self, idx):
#         # Retrieve the input and target
#         inputs = self.inputs[idx]
#         targets = self.targets[idx]

#         # Tokenize the SMILES string
#         tokenized_smiles = self.tokenizer(
#             self.smiles_list[idx],
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"  # Returns a PyTorch tensor
#         )["input_ids"].squeeze(0)  # Remove batch dimension

#         return inputs, tokenized_smiles, targets

# class DynamicDropDataset(Dataset):
#     def __init__(self, inputs, targets, drop_probability=0.0, missing_token=-100):
#         self.inputs = inputs  # Assume data is a NumPy array or a PyTorch tensor
#         self.targets = targets
#         self.drop_probability = drop_probability
#         self.missing_token = missing_token


#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, idx):
#         input1 = self.inputs[idx]
#         target = self.targets[idx]
#         mask = torch.rand_like(input1, dtype=torch.float) < self.drop_probability
#         input1[mask] = self.missing_token
#         return input1, target


class ScheduledOptim():
    # A simple wrapper class for learning rate scheduling
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps, n_current_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = n_current_steps

    def step(self):
        self._optimizer.step()

    def update_lr(self):
        self._update_learning_rate()

    def step_and_update_lr(self):
        # Step with the inner optimizer
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # Zero out the gradients with the inner optimizer
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)) / (n_warmup_steps ** -0.5)
        # return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        # Learning rate scheduling per step
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def get_final_steps(self):
        return self.n_steps

    def get_current_lr(self):
        # Access the learning rate from the first parameter group
        return self._optimizer.param_groups[0]['lr']


def load_loss_function(config):
    if config["custom_loss"]:
        # Dynamically load the custom loss function
        loss_module = importlib.import_module(config["custom_loss_mod"])
        loss_class = getattr(loss_module, config["loss_function"])
        loss_function = loss_class()
    else:
        # Default to MSE or MAE
        if config["loss_function"] == "mse":
            loss_function = nn.MSELoss()
        elif config["loss_function"] == "mae":
            loss_function = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {config['loss']}")
    return loss_function


def train_and_validate(config, model, optimizer, loss_function, loss_tracker, train_loader, val_loader, device):
    model.to(device)  # Move model to the appropriate device
    predictions = evaluate(val_loader, model, loss_function, loss_tracker, device)
    return predictions


def evaluate(dataloader, model, loss_function, loss_tracker, device):
    model.eval()
    total_loss = 0
    data_iter = iter(dataloader)
    input_1, target_1 = next(data_iter)
    # print(input_1.shape)
    loss_tracks = torch.zeros(1, target_1.shape[1]).to(device)
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            targets = targets.squeeze()
            predictions = predictions.squeeze()
            all_predictions.append(predictions)
    out_pred = torch.cat(all_predictions, dim=0)
            # loss_tracks += loss_tracker(predictions, targets)
    return out_pred.to('cpu').numpy()

def train_epoch(dataloader, model, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    for inputs, smiles, targets in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        smiles = smiles.to(device)
        optimizer.zero_grad()
        predictions = model(inputs, smiles)
        targets = targets.squeeze()
        predictions = predictions.squeeze()
        # print(f'{targets.shape=}')
        # print(f'{predictions.shape=}')
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_epoch(dataloader, model, loss_function, loss_tracker, device):
    model.eval()
    total_loss = 0
    data_iter = iter(dataloader)
    input_1, smiles_1, target_1 = next(data_iter)
    # print(input_1.shape)
    loss_tracks = torch.zeros(1, target_1.shape[1]).to(device)
    with torch.no_grad():
        for inputs, smiles, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            smiles = smiles.to(device)
            predictions = model(inputs, smiles)
            targets = targets.squeeze()
            predictions = predictions.squeeze()
            loss = 0
            loss = loss_function(predictions, targets)
            total_loss += loss.item()

            loss_tracks += loss_tracker(predictions, targets)
    return total_loss / len(dataloader), loss_tracks / len(dataloader)


def tr_curve(my_str, data_path):
    with open(data_path, 'a') as f:
        f.write('\n')
        f.write(my_str)


def load_config(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


'''
prop_dict = {
    1: "compound_complexity",
    2: "num_h_bond_acceptors",
    3: "num_h_bond_donors",
    4: "log_p_xlogp3_aa",
    5: "topological_polar_surface_area",
    6: "dipole_moment",
    7: "total_energy",
    8: "total_enthalpy",
    9: "total_free_energy",
    10: "homo_lumo_gap",
    11: "heat_capacity",
    12: "entropy",
    13: "vertical_ip",
    14: "vertical_ea",
    15: "gei",
    16: "max_esp",
    17: "min_esp",
    18: "avg_esp",
    19: "g_solv_octanol",
    20: "total_sasa_octanol",
    21: "g_solv_water",
    22: "total_sasa_water"
}
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to load config and run main.")
    parser.add_argument('config_path', type=str,
                        help="Path to the configuration JSON file")
    args = parser.parse_args()
    config = load_config(args.config_path)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    config["working_directory"] = script_directory
    config["data_path_val"] = "large_property_test.csv"
    main(config)
