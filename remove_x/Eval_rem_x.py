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
import seaborn as sns
import matplotlib.pyplot as plt
from collections.abc import Iterable

def main(config):
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    model_epochs = config['model_epochs']
    checkpoint_fp = os.path.join(config['working_directory'], config['model_save_prefix']) + f'_epoch_{model_epochs-1}.pt'

    if config['load_ex_model'] and config['model_epochs'] == 0:
        print('Caution: You have elected to load an existing model to train (load_ex_model == True) but have set model_epochs to zero.  Double check before proceeding.')
    if not config['load_ex_model'] and config['model_epochs'] != 0:
        sys.exit("Error: If you have a nonzero model_steps argument (meaning you want to continue training an existing model), you also need to change load_ex_model to True.")
    
    model_suffixes = ['_cp_overall.pt'] + [f'_cp_{i}.pt' for i in range(config['num_properties_removed'])]
    # Load and preprocess data
    train_x, train_y, val_x, val_y = preprocess_data(config)
    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(val_y.shape)
    # Setup DataLoader
    train_loader, val_loader = setup_data_loaders(train_x, train_y, val_x, val_y, config['batch_size'])
    print(len(train_loader), len(val_loader))
    
    # Import Model Class
    model_module = importlib.import_module(config['model_file'])
    model_class = getattr(model_module, config['model_class'])

 
    # Instantiate the model and load existing model if applicable
    if config['include_quad']:
        model = model_class(pred_prop=config['properties_removed'])
        summary(model, (28,))
    else:
        model = model_class(pred_prop=config['properties_removed'], input_size=22)
        summary(model, (22,))
  
    # model_suffixes = ['_cp_overall.pt'] + [f'_cp_{i}.pt' for i in range(config['num_properties_removed'])]
    model_suffixes = [f'_cp_{i}.pt' for i in range(config['num_properties_removed'])]
    for _, model_suffix in enumerate(model_suffixes):
        checkpoint_fp = os.path.join(config['working_directory'], f"{config['model_save_prefix']}") + model_suffix
        model.load_state_dict(torch.load(checkpoint_fp))
        data = eval(val_loader, model, device)
        targets = val_y[:,_].numpy()
        # print(data.shape)
        # print(targets.shape)
        if len(model_suffixes) > 1:
            predictions = data[:,_]
        else:
            predictions = data
        with open(os.path.join(config['working_directory'], config['model_save_prefix']) + '_r2.txt', 'a') as f:
            f.write(f' {r_squared(targets, predictions)},')
            print(f' {r_squared(targets, predictions)},')
    with open(os.path.join(config['working_directory'], config['model_save_prefix']) + '_r2.txt', 'a') as f:
        f.write(f'\n')
        for i in model_suffixes:
            f.write('1000,')




def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

def eval(dataloader, model, device):
    model.to(device)
    model.eval()
    all_predictions = []

    with torch.no_grad():
        count = 0
        for inputs, _ in tqdm(dataloader, desc="Validation"):
            count += 1
            if count > 11:
                pass # break
            inputs = inputs.to(device)
            predictions = model(inputs).squeeze()  # Shape: [batch_size, n]
            # print(_, predictions)
            # exit()
            all_predictions.append(predictions.cpu().numpy())  # Move to CPU and convert to numpy

    # Concatenate predictions to form [num_data_points, n]
    all_predictions = np.concatenate(all_predictions, axis=0)
    return all_predictions


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

    # Extract the target property (y) from the data
    prop_index = [prop-1 for prop in config['properties_predicted']]
    print(prop_index)
    train_y = np.zeros((train_comb.shape[0], len(prop_index)))
    val_y = np.zeros((val_comb.shape[0], len(prop_index)))
    train_y[:, :] = train_comb[:, prop_index]
    val_y[:, :] = val_comb[:, prop_index]
    # Prepare the x data with properties_removed filled with the token (-100)
    properties_removed = [p - 1 for p in config['properties_removed']]  # Convert to zero-based indexing
    token = -100

    if config['include_quad']:
        train_x = train_comb.copy()
        val_x = val_comb.copy()
    else:
        # Include only the first 22 properties
        train_x = train_comb[:, :22].copy()
        val_x = val_comb[:, :22].copy()
        properties_removed = [p for p in properties_removed if p < 22]

    # Replace the removed properties with the token
    train_x[:, properties_removed] = token
    val_x[:, properties_removed] = token

    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    val_x = torch.Tensor(val_x)
    val_y = torch.Tensor(val_y)
    return train_x, train_y, val_x, val_y 

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

def setup_data_loaders(train_x, train_y, val_x, val_y, batch_size):
    train_dataset = CustomDataset(train_x, train_y)
    val_dataset = CustomDataset(val_x, val_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


# Custom Dataset to Handle Output Channels as key:value (str:tensor) Pairs
# This may need to be changed depending on your specific application
# Should work fine for single x and y tensors with first index for data point index
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
    epochs = config['epochs']
    current_epochs = config['model_epochs']
    wk_dir = config['working_directory']
    lc_fp = config['loss_curve_fp']
    md_sv_pre = config['model_save_prefix']
    save_every = config['save_every']
    num_prop_rem = config['num_properties_removed']
    patience = config['patience']
    current_patience = 0

    tr_curve(f"epochs, train_loss, val_loss, " + ", ".join(map(str, config['properties_predicted'])), os.path.join(wk_dir, lc_fp))
    model.to(device)  # Move model to the appropriate device
    for epoch in range(current_epochs, current_epochs+epochs):
        # print(f"Epoch {epoch + 1}\n-------------------------------")
        if config['var_lr']:
            optimizer.update_lr()
            print(optimizer.get_current_lr())
        train_loss = train_epoch(train_loader, model, optimizer, loss_function, device)
        val_loss, prop_mses = validate_epoch(val_loader, model, loss_function, loss_tracker, device)
        # prop_mses = prop_mses.squeeze().cpu().numpy()
        tr_curve(f"{epoch+1}, {train_loss}, {val_loss}, " + ", ".join(map(str, prop_mses.squeeze(0).tolist())), os.path.join(wk_dir, lc_fp))
        prop_mses = prop_mses.squeeze().cpu().numpy()
        if isinstance(prop_mses, np.ndarray) and prop_mses.ndim == 0:
            prop_mses = [prop_mses.item()]
        print(f'Completed Epoch {epoch} with val loss {val_loss} and patience {current_patience}.')
        pat_test = False
        if epoch ==  0:
            pat_test = True
            least_val_loss = val_loss
            least_mse = prop_mses
            # if isinstance(least_mse, np.ndarray) and least_mse.ndim == 0:
            #     least_mse = [least_mse.item()]
            torch.save(model.state_dict(), os.path.join(wk_dir, md_sv_pre) + f'_cp_overall.pt')
            for i in range(num_prop_rem):
                torch.save(model.state_dict(), os.path.join(wk_dir, md_sv_pre) + f'_cp_{i}.pt')
        else:
            for i in range(num_prop_rem):
                # print(prop_mses.shape)
                # print(least_mse)
                if prop_mses[i] < least_mse[i]:
                    pat_test = True
                    print(f'        Saving index {i} at epoch {epoch}; loss = {prop_mses[i]}')
                    least_mse[i] = prop_mses[i]
                    torch.save(model.state_dict(), os.path.join(wk_dir, md_sv_pre) + f'_cp_{i}.pt')
            if val_loss < least_val_loss:
                pat_test = True
                print(f'        Saving overall at epoch {epoch}; loss = {val_loss}')
                least_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(wk_dir, md_sv_pre) + f'_cp_overall.pt')
        # print(least_mse)
        print(f'{(isinstance(least_mse, np.ndarray) and least_mse.ndim == 0)=}')
        # if isinstance(least_mse, np.ndarray) and least_mse.ndim == 0:
        #     least_mse = [least_mse.item()]
        # form_list = [f'{x:.3f}' for x in least_mse]
        # print(f"    Lowest Overall: {least_val_loss:.3f}; lowest prop losses: {[err for err in form_list]}")
        print(f"    Lowest Overall: {least_val_loss:.3f}; lowest prop losses: {', '.join(f'{x:.3f}' for x in least_mse)}")
        # print(f'    Lowest Overall: {least_val_loss}; lowest prop losses: {least_mse}')
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

def train_epoch(dataloader, model, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
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
    parser = argparse.ArgumentParser(description="Script to load config and run main.")
    parser.add_argument('config_path', type=str, help="Path to the configuration JSON file")
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)

