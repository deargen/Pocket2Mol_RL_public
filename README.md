# Pocket2Mol_RL_public
Public repository of the paper <Fine-tuning Pocket-conditioned 3D Molecule Generation via Reinforcment Learning>

# License
This work © 2024 by (redacted for blind peer review) is licensed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1). 
Please keep the `LICENSE` file and the current section of `README.md` as they are. 

Note that a provisional patent is under review, which potentially limits the patentability of related ideas. 

# How to reproduce the results of the paper 

## Reinforcement Learning
We do not provide the code for training models.

## Environment Setup
You can set up the environment using the Conda environment file `environment.yml` and install the package from the current repository.
We will provide the docker image in final version.

### Setting Up Environment from Conda Environment File
1. **Create the Conda Environment:**
   Execute the following command to create the Conda environment:

    ```bash
    conda env create -f environment.yml
    ```

2. **Activate the Conda Environment:**
   Once the environment is successfully created, activate it using:

    ```bash
    conda activate pocket2mol_rl
    ```

    Replace `pocket2mol_rl` with the name of your environment.

3. Install Openbabel manually
    Openbabel is a dependency to compute docking scores.
    Because the `openbabel` package is not available in the conda environment, you need to install it manually. 
    ```bash
    bash scripts/install_openbabel.sh
    ```

### Finally) Install the Current Repository

After setting up your environment, proceed with the repository installation. Run the following command in your terminal or VS Code terminal to install the Python package located in the current directory:

```bash
pip install .
```
This installs the package (along with any dependencies specified) into the Python environment inside your Docker or dev container setup. 

## Preparing inference data

We provide the inference data from all baseline models and the proposed model.
[pdb_dict](pdb_dict.json) contains the information of the receptor for each index.
```
bash scripts/download_data.sh
```
FLAG(Fragment based model) failed to generate SDF file for 14, 77 target.

## Sampling 

To reproduce the contents of `test_outputs/*/p2mrl_SDF`, run the following. Prerequisites: `test_outputs/*/receptor.pdb`.

```
bash scripts/p2m_rl_inference_on_test_set.sh
```

## Evaluation 

To reproduce the contents of `test_cache/test_*_outputs`, run the following. Prerequisites: `test_outputs/*/*_SDF` and `test_cache/ref_docking_score.pkl`.

```
scripts/compute_metrics.sh
```
Note that there are unhandled sources of randomness regarding the computation of docking scores etc., which might result in slightly different outcomes from those in the paper. 

To obtain the summary statistics that appear in the paper, run `summarize_results.ipynb`.

## Datasets used in the paper

You don't need to download the datasets to reproduce the results of the paper.

Please refer to [README.md](data/README.md) in the data folder.