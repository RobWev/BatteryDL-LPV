# BatteryDL-LPV
BatteryDL-LPV provides code for identifying nonlinear battery models using a deep-learning-based Linear Parameter-Varying (LPV) framework. Built using deepSI, the approach combines neural-network-based state estimation with an interpretable LPV state-space representation to model battery terminal-voltage dynamics under varying operating conditions.

The repository includes tools for data processing, model training, simulation, and evaluation across different battery chemistries and input-output configurations. The models can incorporate current, accumulated current, overpotential, and temperature information. Data and conventional battery models from PyBatteryID are used for benchmarking and comparison.

# Usage
To use the code, clone this repository and install the required packages including deepSI and PyBatteryID. Next, download the complete "data" folder from the PyBatteryID GitHub repository and place it in the same directory as the scripts in this repository.

The additional data files included in the "extra_data" folder must then be copied into "data/nmc_temp". These files are required for this project but are not available in the PyBatteryID repository.
