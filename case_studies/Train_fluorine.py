import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,  Dataset
from tqdm import tqdm
import h5py
import argparse
import json
import importlib
from torchsummary import summary
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from collections.abc import Iterable
import pandas as pd
from transformers import AutoTokenizer, AutoModel

def main(config):
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    checkpoint_fp = os.path.join(config['working_directory'], config['model_load_path'])

    if config['load_ex_model'] and config['model_epochs'] == 0:
        print('Caution: You have elected to load an existing model to train (load_ex_model == True) but have set model_epochs to zero.  Double check before proceeding.')
    if not config['load_ex_model'] and config['model_epochs'] != 0:
        sys.exit("Error: If you have a nonzero model_steps argument (meaning you want to continue training an existing model), you also need to change load_ex_model to True.")

    # Load and preprocess data
    props = preprocess_data(config)
    print(props.shape)

    props_train = props[:139910]
    props_val = props[139910:]

    # IF USING CHO DATASET, USE 176261 INSTEAD OF 139910 BELOW!!!
    props_masked, mask = perm_mask(props) # mask is 1 if masked, otherwise 0
    props_masked_train = props_masked[:139910]
    props_masked_val = props_masked[139910:]
    mask_train = mask[:139910]
    mask_val = mask[139910:]
    # Setup DataLoader
    train_loader, val_loader_split, val_loader_true = setup_data_loaders(props_train, props_masked_train, mask_train, props_val, props_masked_val, mask_val, config['batch_size'])
    print(len(train_loader), len(val_loader_split), len(val_loader_true))
    # exit()
    # Import Model Class
    model_module = importlib.import_module(config['model_file'])
    model_class = getattr(model_module, config['model_class'])

 
    # Instantiate the model and load existing model if applicable
    if config['include_quad']:
        model = model_class()
        # summary(model, (28,))
    else:
        model = model_class(include_quad=False)
        # summary(model, (22,))
  
    if config['load_ex_model']:
        checkpoint_fp = os.path.join(config['working_directory'], config['model_load_path'])
        model.load_state_dict(torch.load(checkpoint_fp))

    # Load Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=config['max_lr'])
    if config['var_lr']:
        sched = ScheduledOptim(optimizer, lr_mul=config['max_lr'], d_model=64, n_warmup_steps=config['warmup_steps'], n_current_steps=config['model_epochs'])
    else:
        sched = optimizer
    loss_function = load_loss_function(config)
    loss_tracker = PropertyMSE()    

    # Execute Training and Validation
    train_and_validate(
        config,
        model, 
        sched, 
        loss_function, 
        loss_tracker,
        train_loader, 
        val_loader_split,
        val_loader_true,
        device='cuda',
    )



def perm_mask(props):
    shape = props.shape
    np.random.seed(42)  # Set seed for reproducibility
    
    # Generate the binary mask
    binary_mask = (np.random.rand(*shape) < 0.15).astype(int) # 1 is masked, 0 if left alone
    prop_masked = -100 * binary_mask + props * (1-binary_mask)
    return prop_masked, binary_mask    

def preprocess_data(config):

    # Load the CSV file into a DataFrame
    data = pd.read_csv(os.path.join(config['working_directory'], config['data_path']))

    # Extract the SMILES and property data for the 'train' split
    train_data = data[data['split'] == 'train']
    train_properties = train_data.drop(columns=['smiles', 'split']).to_numpy()

    # Extract the SMILES and property data for the 'val' split
    val_data = data[data['split'] == 'val']
    val_properties = val_data.drop(columns=['smiles', 'split']).to_numpy()
    print(val_properties.shape)
   

    # train_comb = np.loadtxt(os.path.join(config['working_directory'], config['data_path_train']), delimiter=',')
    # val_comb = np.loadtxt(os.path.join(config['working_directory'], config['data_path_val']), delimiter=',')
    print(train_properties.shape) 
    print(val_properties.shape)
    condition_train = (np.abs(train_properties[:, 18]) <= 0.2) & (np.abs(train_properties[:, 20]) <= 0.2)
    train_properties = train_properties[condition_train]
    print(train_properties.shape)



    condition_val = (np.abs(val_properties[:, 18]) <= 0.2) & (np.abs(val_properties[:, 20]) <= 0.2)
    val_properties = val_properties[condition_val]
    print(val_properties.shape)


    if config['rem_outliers']:
        # Load property bounds
        with open(os.path.join(config['working_directory'], config['outlier_bound_fp']), 'r') as f:
            bounds = json.load(f)

        # Remove outliers based on bounds
        for prop, bound in bounds.items():
            prop_idx = int(prop)
            min_val, max_val = bound['min'], bound['max']
            train_test = (train_properties[:, prop_idx] >= min_val) & (train_properties[:, prop_idx] <= max_val)
            train_properties = train_properties[train_test]
            val_test = (val_properties[:, prop_idx] >= min_val) & (val_properties[:, prop_idx] <= max_val)
            val_properties = val_properties[val_test]

    print(f"Shape after removing outliers: Train = {train_properties.shape}, Validation = {val_properties.shape}")


    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 100))

    # Fit the scaler to the data and transform it
    scaler = joblib.load('scaler_0_100_after_outliers.pkl')
    train_properties = scaler.transform(train_properties)
    val_properties = scaler.transform(val_properties)

    # Make sure no validation values interfere with -100 missing token.
    assert np.all(val_properties > -99), "Error: Validation data contains values less than -99! Will interfere with missing tokens!"

    # Warning for values less than -10
    rows_with_ood = np.any(val_properties < -10, axis=1)  # Boolean array indicating rows with values < -10
    num_ood_rows = np.sum(rows_with_ood)  # Count the number of such rows
    if num_ood_rows > 0:
        print(f"WARNING! You have {num_ood_rows} validation data points that are out of distribution.")
    
    ''' 
    # **Action 3: Compute and print the min-max bounds for each property**
    property_bounds = {
        f"Property {i}": (np.min(val_comb[:, i]), np.max(val_comb[:, i]))
        for i in range(val_comb.shape[1])
    }
    for prop, bounds in property_bounds.items():
        print(f"{prop}: Min = {bounds[0]:.2f}, Max = {bounds[1]:.2f}")
    exit()
    '''

    ''' 
    # **Action 3: Compute and print the min-max bounds for each property**
    property_bounds = {
        f"Property {i}": (np.min(train_comb[:, i]), np.max(train_comb[:, i]))
        for i in range(train_comb.shape[1])
    }
    for prop, bounds in property_bounds.items():
        print(f"{prop}: Min = {bounds[0]:.2f}, Max = {bounds[1]:.2f}")
    exit()
    '''


    if config['include_quad']:
        train_x = train_properties.copy()
        val_x = val_properties.copy()
    else:
        # Include only the first 22 properties
        train_x = train_properties[:, :22].copy()
        val_x = val_properties[:, :22].copy()
    print(f'{train_x.shape=}')
    print(f'{val_x.shape=}')
    props = np.concatenate((train_x, val_x), axis=0)

    return props



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
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} and targets {targets.shape}")

        # Compute MSE along dim=0 (property-wise across the batch)
        mse_per_property = torch.mean((predictions - targets) ** 2, dim=0)
        return mse_per_property

def setup_data_loaders(props_train, props_masked_train, mask_train, props_val, props_masked_val, mask_val, batch_size):
    props_train = torch.Tensor(props_train)
    props_masked_train = torch.Tensor(props_masked_train)
    mask_train = torch.tensor(mask_train, dtype=torch.bool)
    props_val = torch.Tensor(props_val)
    props_masked_val = torch.Tensor(props_masked_val)
    mask_val = torch.tensor(mask_val, dtype=torch.bool)
    train_dataset = SMILESTrainDataset(props_masked_train, mask_train)
    val_dataset_split = SMILESTrainDataset(props_masked_val, mask_val)
    train_dataset[16004]
    val_dataset_true = SMILESValDataset(props_masked_val, props_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader_split = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader_true = DataLoader(val_dataset_true, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, val_loader_split, val_loader_true


# Custom Dataset to Handle Output Channels as key:value (str:tensor) Pairs
# This may need to be changed depending on your specific application
# Should work fine for single x and y tensors with first index for data point index

'''
class CustomDataset(Dataset):
    def __init__(self, inputs,  targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        target = self.targets[idx]
        return inputs, target
'''

class SMILESTrainDataset(Dataset):
    def __init__(self, props_masked, mask, frac_add_mask=0.15, max_length=128):
        self.props_masked = props_masked                # Corresponding input data
        self.mask = mask
        self.max_length = max_length
        self.frac_add_mask = frac_add_mask

    def __len__(self):
        return len(self.props_masked)

    def __getitem__(self, idx):
        # Retrieve the input and target
        targets = self.props_masked[idx]
        masks = self.mask[idx]
        # Determine the unmasked positions in targets
        unmasked_positions = torch.logical_not(masks)  # True where values are unmasked

        # Generate a random mask for additional masking
        new_mask = torch.zeros_like(masks, dtype=torch.bool)
        new_mask[masks] = 1
        newish_mask = torch.rand_like(masks, dtype=torch.float) < self.frac_add_mask
        new_mask[unmasked_positions] = newish_mask[unmasked_positions]
      
        # Update the mask by combining the original and additional masks
        # additional_mask = random_mask.bool()  # Convert to boolean mask
        # combined_mask = masks | additional_mask  # Logical OR to combine masks

        # Create the inputs by applying the combined mask to the targets
        inputs = targets * (torch.logical_not(new_mask)) + new_mask * (-100)

        return inputs, targets, masks

class SMILESValDataset(Dataset):
    def __init__(self, inputs, targets, max_length=128):
        self.inputs = inputs                # Corresponding input data
        self.targets = targets              # Corresponding target data
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Retrieve the input and target
        inputs = self.inputs[idx]
        targets = self.targets[idx]

        return inputs, targets


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


def train_and_validate(config, model, optimizer, loss_function, loss_tracker, train_loader, val_loader_split, val_loader_true, device):
    epochs = config['epochs']
    current_epochs = config['model_epochs']
    wk_dir = config['working_directory']
    lc_fp = config['loss_curve_fp']
    md_sv_pre = config['model_save_prefix']
    save_every = config['save_every']
    patience = config['patience']
    current_patience = 0

    tr_curve(f"epochs, train_loss, val_loss_split, val_loss_true", os.path.join(wk_dir, lc_fp))
    model.to(device)  # Move model to the appropriate device
    for epoch in range(current_epochs, current_epochs+epochs):
        # print(f"Epoch {epoch + 1}\n-------------------------------")
        if config['var_lr']:
            optimizer.update_lr()
            print(optimizer.get_current_lr())
        train_loss = train_epoch(train_loader, model, optimizer, loss_function, device)
        val_loss_split = validate_epoch_split(val_loader_split, model, loss_function, loss_tracker, device)
        val_loss_true = validate_epoch_true(val_loader_true, model, loss_function, loss_tracker, device)
        tr_curve(f"{epoch+1}, {train_loss}, {val_loss_split, val_loss_true}", os.path.join(wk_dir, lc_fp))
        print(f'Completed Epoch {epoch} with val loss split {val_loss_split:.6f} and val loss true {val_loss_true:.6f} and patience {current_patience}.')
        pat_test = False
        if epoch ==  0:
            pat_test = True
            least_val_loss = val_loss_split
            torch.save(model.state_dict(), os.path.join(wk_dir, md_sv_pre) + f'_cp_best.pt')
        else:
            if val_loss_split < least_val_loss:
                pat_test = True
                print(f'        Saving overall at epoch {epoch}; loss_split = {val_loss_split:.6f}; loss_true = {val_loss_true:.6f}')
                least_val_loss = val_loss_split
                torch.save(model.state_dict(), os.path.join(wk_dir, md_sv_pre) + f'_cp_best.pt')
        print(f"    Lowest Overall: {least_val_loss:.6f}")
        if pat_test:
            current_patience = 0
        else:
            current_patience += 1
        if current_patience >= patience:
            break
        '''
        if epoch % save_every == save_every-1:
            torch.save(model.state_dict(), os.path.join(wk_dir, md_sv_pre) + f'_epoch_{epoch}.pt')
        '''
    torch.save(model.state_dict(), os.path.join(wk_dir, md_sv_pre) + f'_cp_end.pt')

def train_epoch(dataloader, model, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    for inputs, targets, mask in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        targets = targets.squeeze()
        predictions = predictions.squeeze()
        loss = loss_function(predictions, targets, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_epoch_split(dataloader, model, loss_function, loss_tracker, device):
    model.eval()
    total_loss = 0
    data_iter = iter(dataloader)
    input_1, target_1, mask_1 = next(data_iter)
    loss_tracks = torch.zeros(1, target_1.shape[1]).to(device)
    with torch.no_grad():
        for inputs, targets, mask in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            predictions = model(inputs)
            targets = targets.squeeze()
            predictions = predictions.squeeze()
            loss = 0
            loss = loss_function(predictions, targets, mask)
            total_loss += loss.item()

            # loss_tracks += loss_tracker(predictions, targets)
    return total_loss / len(dataloader)


def validate_epoch_true(dataloader, model, loss_function, loss_tracker, device):
    model.eval()
    total_loss = 0
    data_iter = iter(dataloader)
    input_1, target_1 = next(data_iter)
    # print(input_1.shape)
    loss_tracks = torch.zeros(1, target_1.shape[1]).to(device)
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            targets = targets.squeeze()
            predictions = predictions.squeeze()
            loss = 0
            loss = loss_function(predictions, targets)
            total_loss += loss.item()
        
            # loss_tracks += loss_tracker(predictions, targets)
    return total_loss / len(dataloader)


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
    parser = argparse.ArgumentParser(description="Script to load config and run main.")
    parser.add_argument('config_path', type=str, help="Path to the configuration JSON file")
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)

