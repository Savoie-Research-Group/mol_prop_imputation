# mol_prop_imputation
Implementation of transformer-based framework for generalized molecular property imputation.  Includes support for property and structure guided imputation under both fixed and variable missingness regimes.  Can be adapted to non-molecular imputation tasks.



- **How to run (all tasks)**:
  - Train: `python Train_xxx.py config_xxx.json`
  - Eval: `python Eval_xxx.py config_xxx.json`
  - Note: It is preferable to have all files in a common directory (eg. data, model, training script, checkpoints, etc.) (cf. 'Config fields')
- **Data & Checkpoints**:
  - Datasets for each task are located in zenodo repositories here: https://zenodo.org/records/17519032 .  The .csv file for the two 'random_drop' tasks is located in the repository for 'struct_prop_comparison'.
  - Select checkpoints are shared through the same zenodo repositories.  The Evaluation scripts infer model checkpoints from config arguments, so it is best not to modify the file names.  All checkpoints for all models are available upon request.
  - In preprocessing, outliers are generally removed according to per-property bounds in `outlier_bounds.json`.
- **Missingness handling (shared)**:
  - Property inputs use a custom embedding that mixes a learned token for missing values with a linear projection for observed values. Several regimes also introduce random masking during training to simulate missingness.
- **Checkpoints and logs (shared)**:
  - Training writes loss curves to `loss_curve_fp`. Best checkpoints are saved as `{model_save_prefix}_cp_overall.pt` and, when predicting multiple properties, per-property bests `{model_save_prefix}_cp_{i}.pt`. Early stopping uses `patience`.

### Config fields
- **Paths and execution**
  - `working_directory`: Base path for data and artifacts. Many scripts join this with other filenames; prefer running from the folder that contains both the script and the files declared in the config.  
  - `model_file`, `model_class`: Module or file path and class name to import the model. `model_file` is the filepath in which the model architecture is stored.  `model_class` is the name of the model architecture class within that file.
  - `model_save_prefix`, `loss_curve_fp`: Prefix for saved checkpoints/metrics; plain filename for the loss curve text file.
  - Note: Arguments for some tasks take absolute paths while others do not.  Look at the provided config files to see whether absolute or relative paths are needed.
- **Optimization**
  - `batch_size`, `epochs`, `model_epochs` (starting epoch), `max_lr`, `var_lr`, `warmup_steps`, `patience`.
  - `var_lr: true` enables a Transformer-style schedule via `ScheduledOptim` with warmup; otherwise a fixed Adam optimizer is used.
- **Data and preprocessing**
  - `rem_outliers` and `outlier_bound_fp`: Apply property-specific bounds before scaling.
  - Scaling: MinMax scaling to 0–100. Some scripts fit a scaler according to `outlier_bound_fp`; others load a precomputed scaler (`scaler_0_100_after_outliers.pkl`).
  - `include_quad`: If true, the last 6 fields (quadrupole vector) are included and embedded as an extra token in the model. If this code is ever adapted to non-molecular tasks, this functionality will need to be removed.
- **Loss function**
  - `custom_loss`: a boolean of whether a custom loss is required ('mse' or 'mae' are built in options) 
  - `custom_loss_mod`, `loss_function`: If `custom_loss`==true, `custom_loss_mod` is the filepath containing the loss function and `loss_function` is the name of the class or function of the loss function within that file (cf. `model_file` and `model_class`).  Otherwise, `custom_loss_mod` can be left blank and `loss_function` populated with 'mae' or 'mse'.  
  - A custom loss function (i.e., `masked_loss.MaskedLoss`) was only used in the case study because 'truly missing' data was permanently masked and needed to be omitted in calculating the loss.  In all other cases, 'mse' was used.
- **Task-specific**
  - `num_properties_removed`, `properties_removed`, `properties_predicted`: Define which properties are removed from inputs and which are predicted.
  - `fraction_dropped`: Controls dynamic random masking during training in the random-drop regimes.

### Model architecture overview (shared across regimes)
- **Property encoder and missing-token handling**:
  - `CustomEmbedding`: Per-property linear layers map scalar inputs; a learned per-property token is mixed in when the input equals `-100` (missing). When `include_quad` is true, the 6-D quadrupole is projected and appended as an additional token.
- **Attention backbone and decoders**:
  - A Transformer encoder (`batch_first=True`) processes the property-token sequence (plus the quadrupole token if enabled). Property-specific decoders are linear heads either per property (full-width) or limited to the predicted subset (remove_x, mol2prop select indices).
- **Structure integration (ChemBERTa)**
  - In structure-containing regimes, tokenized SMILES are passed to a frozen ChemBERTa (`DeepChem/ChemBERTa-77M-MLM`). The [CLS] embedding is projected to `embed_dim` and concatenated as an extra 'property' before the Transformer encoder. Structure-only models pass SMILES through ChemBERTa and a MLP to predict properties directly.

### Folder-by-folder details

#### remove_x
- **Purpose**: Systematic evaluations where specified properties are removed from inputs and predicted explicitly.
- **Data**: Takes separate data files for individual split (eg. `large_property_train.csv`).   

#### struct_prop_comp
- **Purpose**: Side-by-side comparison of structure-only, property+structure (combined), and the config conventions used throughout.
- **Scripts**:
  - Structure-only: `Train_mol2prop.py` with `config_str_ex.json` and `model_mol2prop.py`.
  - Combined: `Train_all2prop.py` with `config_all_ex.json` and `model_14.py`. 
- **Data**: A combined file with all splits: `large_property_combined.csv`

#### random_drop_prop2prop
- **Purpose**: Property-only imputation with dynamic random missingness; learn to infer properties from other properties.
- **Data**: uses separate train/val CSVs (`large_property_train.csv`, `large_property_val.csv`) available in the `remove_x` Zenodo repository.

#### random_drop_all2prop
- **Purpose**: Combined structure + properties regime with dynamic random masking on properties; learn jointly from property tokens and SMILES.
- **Data**: a single combined CSV with SMILES (`large_property_combined.csv`) is available in the `struct_prop_comparison` Zenodo repository.

#### case_studies
- **Purpose**: Demonstrate self-supervised imputation for dataset subgroups (e.g., CHO, fluorine) in which some data is 'truly missing'.




### Practical Notes
- **Run location and paths**: Prefer running from within the specific folder so that relative paths in configs resolve against `working_directory`. Several eval scripts also override some config paths at runtime to files in their own directory.


