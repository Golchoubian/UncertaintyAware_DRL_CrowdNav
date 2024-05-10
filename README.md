# UncertaintyAware_DRL_CrowdNav

This repository contains the code for our paper titled "Uncertainty-Aware DRL for Autonomous Vehicle Crowd Navigation in Shared Space". For more detailes please refer to our [paper]. A video of the simulation results is also provided [here].


<div style=""display: block; margin: 0 auto; text-align: center;"">
    <img src="https://github.com/Golchoubian/UncertaintyAware_DRL_CrowdNav/blob/main/figures/illustration.png?raw=true" alt="illustration" width="700"> 
</div>


Our method introduces an innovative approach for safe and socially compliant navigation of low-speed autonomous vehicles (AVs) in shared environments with pedestrians. Unlike existing deep reinforcement learning (DRL) algorithms, which often overlook uncertainties in pedestrians' predicted trajectories, our approach integrates prediction and planning while considering these uncertainties. This integration is facilitated by a model-free DRL algorithm trained in a novel simulation environment reflecting realistic pedestrian behavior in shared spaces with vehicles. We employ a novel reward function that encourages the AV to respect pedestrians' personal space, reduce speed during close approaches, and minimize collision probability with their predicted paths.


# Installtion

Create a  virtual environment or conda environmnet using python version 3.9, and Install the required python packages:

```bash
pip install -r requirements.txt
```

Install pytorch version 1.12.1 using the instructions [here](https://pytorch.org/get-started/previous-versions/#v1121):

```bash
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1

```

Install [OpenAI Baselines](https://github.com/openai/baselines#installation)

```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
```

## Overview

There are four main folders within this repository:

* `move_plan/`: Contains the configuration file and policy used for the robot/AV in the simulation environment.

* `ped_pred/`: Includes files for running inference on our pedestrian trajectory prediction model, named PollarCollisionGrid ([PCG](https://github.com/Golchoubian/PolarCollisionGrid-PedestrianTrajectoryPrediction)) and its uncertainty-aware version [UAW-PCG](https://github.com/Golchoubian/PolarCollisionGrid-UncertaintyAware).

* `ped_sim/`: Contatins the files for the simulation environment.

* `rl/`: Holds the files for the DRL policy network, the ppo algorithm, and the wrapper for the prediction network.

* `trained_model/`: Contains the trained models reported in our paper

## Simulation Environment


In our project focusing on crowd navigation for Autonomous Vehicles (AVs) in shared spaces, we have developed a custom simulation environment. Based on the [HBS dataset](https://leopard.tu-braunschweig.de/receive/dbbs_mods_00069907), which captures pedestrians' and vehicles' trajectories in shared spaces, our simulation environment replicates real pedestrian behaviors. This enables effective AV training by providing realistic scenarios. Additionally, we utilize this dataset to train our data-driven pedestrian trajectory prediction module.

From the HBS dataset, we have extracted 310 scenarios corresponding to the vehicles in the dataset, which are divided into training, testing, and validation sets. Pedestrian behaviors are simulated using real-world data, while AV actions are governed by the DRL policy network. These scenarios present dynamic situations where AVs must adapt by decelerating to yield to pedestrians or accelerating to avoid collisions.

The simulation environment being developed, based on real trajectory data, provides the advantage of having human-driven trajectories for each scenario, which can be compared with the trajectories of AVs' trained navigation policy. Integrated into a gym environment, our simulation environment can serve as a benchmark for training AVs in crowd navigation.


## Model training

According to the model you want to train modify the configurations in the following two files, with some of the key parameters mentioned below:

* Environment configurations in `move_plan/configs/config.py`.
  - `sim.predict_method` comes with the following options:
    - `inferred`: When using pedestrians predicted trajecoteis in the DRL algorithm and replying on the PCG/UAW-PCG predictor model to produce the prediction
    - `none`: When not using pedestrians predicted trajecoteis in the DRL algorithm
    - `truth`: When using pedestrians predicted trajecoteis in the DRL algorithm and replying on the ground truth prediction from the dataset for that

* Network configurations in `arguments.py`
  - `env_name` comes with the following options and must be consistant with `sim.predict_method` in `move_plan/configs/config.py`
    - `PedSim-v0`: if using prediction (either PCG/UAW-PCG or ground truth)
    - `PedSimPred-v0`: if not using prediction.
  - `uncertainty_aware` is a boolean argument:
    - `True`: When using the uncertainty-aware polar collision grid prediction model (UAW-PCG)
    - `False`: When using the origianl polar collision grid prediction model (PCG)

  After all adjustments has been made, run:
  ```python
  python train.py
  ```
  
## Model testing

For testing our already trained models in the paper and reproducing the results of table III, adjust the arguments in line 24-31 of the `test.py` file and run:
```python
python test.py
```

The three main arguments to adjust are as follows:
  - `test_model` specifies the name of the trained model to test:
      - `UAW_PCG_pred`: The DRL model trained with UAW-PCG prediction model
      - `PCG_pred`: The DRL model trained with the PCG prediction model
      - `GT_pred`: The DRL model trained with groun truth prediction model
      - `No_pred`: The DRL model trained without any prediction data
      - `No_pred_SD`: The DRL model trained without any prediction data but with a speed dependant danger penalty reward function
      - `Human_Driver`: The trajecotry of the human dirver in the dataset

        Note: the `config.py` and `arguments.py` in the saved models folder will be loaded, instead of those in the root directory. (Therefore, no need to change the config and argument file of the root directory when generating the test result of each provided trained model)
  
  - `test_case` specifies the scenarios in the test set to test the model on 
    - `-1` runs the test on all scenarios in the test set
    - `Any numver in raneg: [248-310]` run the test on the specified scenario numbers in the test set

      Note: Among the 310 extracted scenario from the HBS dataset, here is the scenario numbers within each subdivided category of train, validation and test:
      - validation: scenario numbers: 0 - 48
      - train: scenario numbers: 48 - 247
      - test: scenario numbers: 248 - 310

  - `visualize`, if set to true will visualize the simualtion environment with the gif saved in `traine_models/ColliGrid_predictor/visual/gifs`
      
      Note: the visualization will slow down testing significantly.
  

  <div style="display: flex; justify-content: center;">
  <img src="https://github.com/Golchoubian/UncertaintyAware_DRL_CrowdNav/blob/main/figures/UAW-PCG_pred.gif?raw=true" alt="GIF 1" width="500">
  </div>



## Citation

```bibtex
@inproceedings{golchoubian2023polar,
  title={Polar Collision Grids: Effective Interaction Modelling for Pedestrian Trajectory Prediction in Shared Space Using Collision Checks},
  author={Golchoubian, Mahsa and Ghafurian, Moojan and Dautenhahn, Kerstin and Azad, Nasser Lashgarian},
  booktitle={2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)},
  pages={791--798},
  year={2023},
  organization={IEEE}
}
```


# Acknowledgment

This project is builds upon the codebase from [CrowdNav_Prediction_AttnGraph
repsitory](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)

