# Pocket2Mol_RL_public
Public repository of the paper <Fine-tuning Pocket-conditioned 3D Molecule Generation via Reinforcment Learning>

# License
This work © 2024 by Deargen Inc. is licensed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1). 
Please keep the `LICENSE` file and the current section of `README.md` as they are. 

Note that a provisional patent is under review, which potentially limits the patentability of related ideas. 

# How to reproduce the results of the paper 

## Reinforcement Learning
We do not provide the code for training models.

## Environment Setup
1. This guide outlines the steps to set up your environment using the Docker image `deargen/pocket2mol_rl_public:latest`. You can work with this environment either interactively in a terminal or within a development container in Visual Studio Code.

 2. If you don't want to use Docker, you can set up the environment using the Conda environment file `environment.yml` and install the package from the current repository.

 3. Choose one of the following options to set up your environment and proceed to the finally section to install the current repository.

 ### Option1) Using Docker Interactively
 Open your terminal and execute the command below to start an interactive bash shell within the Docker container:

 ```bash
 docker run -it deargen/pocket2mol_rl_public:latest /bin/bash
 ```

 This will pull the `deargen/pocket2mol_rl_public:latest` image from DockerHub (if not already present locally) and open a bash shell in the container.

 You have to logged in to the DockerHub to pull the image.

 ### Option2) Using Visual Studio Code Dev Containers

 To open a folder in a dev container using Visual Studio Code, ensure you have the Remote - Containers extension installed, then follow these steps:
 - Press `Command` + `Shift` + `P` on macOS (`Ctrl` + `Shift` + `P` on Windows/Linux) to open the command palette.
 - Type and select `Dev Container: Open Folder in Container`.
 - Follow the prompts to select the folder you wish to open in the container.
 - For more details, refer to the [Visual Studio Code documentation on dev containers](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container).

 ### Option3) Setting Up Environment from Conda Environment File
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
    This command installs the openbabel package for the user, independent of any Conda environment.
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
Note that 
- the outputs of the models "ar", "ligan", and "targetdiff" were derived from those of [https://github.com/guanjq/targetdiff](https://github.com/guanjq/targetdiff)
- the outputs of the model "pocket2mol" were produced using the code provided in [https://github.com/pengxingang/Pocket2Mol](https://github.com/pengxingang/Pocket2Mol)
- FLAG(Fragment based model) failed to generate SDF file for 14, 77 target.

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