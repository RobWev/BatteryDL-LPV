import deepSI
from deepSI import System_data, System_data_list
import numpy as np
from pybatteryid import ModelStructure
from data import helper
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from pybatteryid.simulation import simulate_model
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Any
from pybatteryid.utilities import invert_voltage_function, load_model_from_file

# -----------------------------
# Flexible process_data
# -----------------------------
def _flag_enabled(flags: dict[str, Any], *aliases: str) -> bool:
    """Return True if any of the provided aliases evaluates truthy in flags."""
    for alias in aliases:
        if alias in flags and flags[alias]:
            return True
    # case-insensitive fallback
    lowered = {str(k).lower(): v for k, v in flags.items()}
    for alias in aliases:
        key = alias.lower()
        if key in lowered and lowered[key]:
            return True
    return False


def process_data(voltage, current, initial_soc, battery_capacity,
                 emf_interpolate, with_temp=False, temp=None,
                 dV_dT_interpolate=None, T_ref=None,
                 with_hysteresis=False, hysteresis_interpolate=None):
    N = np.size(voltage)
    SoC = np.zeros(N)
    emf_values = np.zeros(N)
    Vop = np.zeros(N)
    I_pol = np.zeros(N)
    I_sum = np.zeros(N)
    
    hysteresis_values = np.zeros(N) if with_hysteresis else None
    V_minus_hyst = np.zeros(N) if with_hysteresis else None

    # Initial values
    SoC[0] = initial_soc
    soc_idx = (np.abs(emf_interpolate[0] - SoC[0])).argmin()
    emf_values[0] = emf_interpolate[1][soc_idx]
    
    if with_temp and temp is not None and dV_dT_interpolate is not None:
        dV_dT = dV_dT_interpolate[1][soc_idx]
        emf_values[0] += dV_dT * (temp[0] - T_ref)

    Vop[0] = voltage[0] - emf_values[0]

    if with_hysteresis and hysteresis_interpolate is not None:
        hyst_val = hysteresis_interpolate[1][soc_idx]
        hysteresis_values[0] = hyst_val
        V_minus_hyst[0] = voltage[0] - hyst_val

    # Initial polarity and current sum
    if current[0] > 0: I_pol[0] = 1
    elif current[0] < 0: I_pol[0] = -1
    else: I_pol[0] = 0
    I_sum[0] = current[0]

    # Loop
    for i in range(1, N):
        SoC[i] = SoC[i-1] + current[i] / battery_capacity
        soc_idx = (np.abs(emf_interpolate[0] - SoC[i])).argmin()
        emf_values[i] = emf_interpolate[1][soc_idx]

        if with_temp and temp is not None and dV_dT_interpolate is not None:
            dV_dT = dV_dT_interpolate[1][soc_idx]
            emf_values[i] += dV_dT * (temp[i] - T_ref)

        Vop[i] = voltage[i] - emf_values[i]

        if with_hysteresis and hysteresis_interpolate is not None:
            hyst_val = hysteresis_interpolate[1][soc_idx]
            hysteresis_values[i] = hyst_val
            V_minus_hyst[i] = voltage[i] - hyst_val

        if current[i] > 0:
            I_pol[i] = 1
        elif current[i] < 0:
            I_pol[i] = -1
        else:
            I_pol[i] = I_pol[i-1]

        I_sum[i] = I_sum[i-1] + current[i]

    if with_hysteresis:
        return SoC, Vop, emf_values, I_pol, I_sum, hysteresis_values, V_minus_hyst
    else:
        return SoC, Vop, emf_values, I_pol, I_sum


def load_cell_data(cell="nmc_temp", flags=None):
    if flags is None:
        flags = {"Vop": False,"SoC_out": False,"temp": True,"SoC": False,"I_sum": False,"SoC_pol": False,"u_abs": False}
    else:
        default_flags = {"Vop": False,"SoC_out": False,"temp": True,"SoC": False,"I_sum": False,"SoC_pol": False,"u_abs": False}
        default_flags.update(flags)
        flags = default_flags

    datasets = {}
    dataset_names = []    

    if cell == "nmc_temp":
        battery_capacity=10344.169
        base_path = "data/nmc_with_temperature"
        emf_data = helper.load_npy_data(f"{base_path}/gitt_data.npy")
        model_structure = ModelStructure(battery_capacity, 1)
        model_structure.add_emf_function({
            'soc_values': emf_data['soc_values'],
            'voltage_values': emf_data['voltage_values'],
            'dVdT_values': emf_data['dVoltage_dTemperature_values'],
            'reference_temperature_value': emf_data['reference_temperature_value']
        })

        # Load training sets 0_6 to 5_6
        for i in range(6):
            name = f"train_{i}_6"
            path = f"{base_path}/identification_data_{i}_6.npy"
            datasets[name] = helper.load_npy_data(path)
            dataset_names.append(name)

        # Load validation sets
        for cond in ["hot", "cold", "normal"]:
            name = f"test_{cond}"
            path = f"{base_path}/validation_data_{cond}.npy"
            datasets[name] = helper.load_npy_data(path)
            dataset_names.append(name)

        with_temp = True
        with_hysteresis = False
        hysteresis_interpolate = None

    elif cell == "nmc":
        battery_capacity=3440.05372
        base_path = "data/nmc"
        emf_data = helper.load_npy_data(f"{base_path}/gitt_data.npy")
        model_structure = ModelStructure(battery_capacity, 1)
        model_structure.add_emf_function({
            'soc_values': emf_data['soc_values'],
            'voltage_values': emf_data['voltage_values']
        })

        datasets["train"] = helper.load_npy_data(f"{base_path}/identification_data.npy")
        datasets["test"] = helper.load_npy_data(f"{base_path}/validation_data.npy")
        datasets["val"] = helper.load_npy_data(f"{base_path}/drive_cycle_data.npy")
        dataset_names = ["train", "test", "val"]

        with_temp = False
        with_hysteresis = False
        hysteresis_interpolate = None

    elif cell == "lfp":
        battery_capacity=13270.180265768407
        base_path = "data/lfp"
        emf_data = helper.load_npy_data(f"{base_path}/quasi_emf.npy")
        hysteresis_data = helper.load_npy_data(f"{base_path}/hysteresis_data.npy")

        model_structure = ModelStructure(battery_capacity, 1)
        model_structure.add_emf_function({
            'soc_values': emf_data['soc_values'],
            'voltage_values': emf_data['voltage_values']
        })

        x = np.linspace(0, 1, 10001)
        hysteresis_interpolate = np.array((x, np.interp(x, hysteresis_data['soc_values'], hysteresis_data['voltage_values'])))

        datasets["train_1"] = helper.load_npy_data(f"{base_path}/identification_data_first.npy")
        datasets["train_2"] = helper.load_npy_data(f"{base_path}/identification_data_second.npy")
        datasets["train_3"] = helper.load_npy_data(f"{base_path}/identification_data_third.npy")
        datasets["test"] = helper.load_npy_data(f"{base_path}/validation_data.npy")
        dataset_names = ["train_1", "train_2", "train_3", "test"]

        with_temp = False
        with_hysteresis = True

    # Build interpolates for emf
    x = np.linspace(0, 1, 10001)
    emf_interpolate = np.array((x, np.interp(x, emf_data['soc_values'], emf_data['voltage_values'])))
    dV_dT_interpolate = None
    T_ref = None
    
    if with_temp:
        dV_dT_interpolate = np.array((x, np.interp(x, emf_data['soc_values'], emf_data['dVoltage_dTemperature_values'])))
        T_ref = emf_data['reference_temperature_value']

    # Assign initial_soc and process each dataset
    sys_data_dict = {}
    results = {}
    u_data = {}
    y_data = {}
    
    if cell == "nmc_temp":
        voltage_name = 'voltage_values'
        current_name = 'current_values'
    else:
        voltage_name = 'voltage'
        current_name = 'current'
    
    for name, dataset in datasets.items():
        if with_temp:
            dataset['initial_soc'] = invert_voltage_function(model_structure.emf_function, dataset[voltage_name][0], dataset['temperature_values'][0])
        else:
            dataset['initial_soc'] = invert_voltage_function(model_structure.emf_function, dataset[voltage_name][0])

        results[name] = process_data(dataset[voltage_name], dataset[current_name], dataset['initial_soc'], battery_capacity,
            emf_interpolate, with_temp=with_temp,temp=dataset['temperature_values'] if with_temp else None,
            dV_dT_interpolate=dV_dT_interpolate, T_ref=T_ref, with_hysteresis=with_hysteresis, hysteresis_interpolate=hysteresis_interpolate)

        flag_sources = {
            "Vop": lambda name, dataset: results[name][1].copy(),
            "SoC_out": lambda name, dataset: results[name][0].copy(),
            "temp": lambda name, dataset: dataset.get("temperature_values", np.array([])).copy(),
            "SoC": lambda name, dataset: results[name][0].copy(),
            "I_sum": lambda name, dataset: results[name][4].copy(),
            "SoC_pol": lambda name, dataset: results[name][3].copy(),
            "u_abs": lambda name, dataset: np.abs(dataset[current_name].copy()),
        }
        
        # --- y_data ---
        flag_vop = _flag_enabled(flags, "Vop", "V_op", "overpotential")

        if flag_vop:
            y_data[name] = flag_sources["Vop"](name, dataset)
        else:
            y_data[name] = dataset[voltage_name].copy()

        if flags.get("SoC_out", False):
            y_data[name] = np.column_stack([y_data[name], flag_sources["SoC_out"](name, dataset)])

        # --- u_data ---
        u_parts = [dataset[current_name].copy()]
        for flag, func in flag_sources.items():
            if flag in ["Vop", "SoC_out"]:
                continue  # already handled in y_data
            if flags.get(flag, False):
                val = func(name, dataset)
                if val.size > 0:
                    u_parts.append(val)

        u_data[name] = np.column_stack(u_parts)

        # Orientation check: ensure (N, nu)
        if u_data[name].shape[0] != len(dataset[voltage_name]):
            u_data[name] = u_data[name].T

        sys_data = deepSI.System_data(u=u_data[name], y=y_data[name])

        # Ensure extra metadata container exists for downstream use
        existing_extra = getattr(sys_data, "extra", None)
        if not isinstance(existing_extra, dict):
            sys_data.extra = {} if existing_extra is None else {"original_extra": existing_extra}
        else:
            sys_data.extra = dict(existing_extra)

        if flag_vop:
            emf_values = results[name][2]
            sys_data.extra["emf_values"] = np.array(emf_values, copy=True)
            sys_data.extra["signal_type"] = "Vop"
        else:
            sys_data.extra.setdefault("signal_type", "voltage")

        sys_data_dict[name] = sys_data

    # Build combined training dataset (list)
    train_comb = System_data_list()
    for key in dataset_names:
        if key.startswith("train") or key.startswith("val"):
            train_comb.append(sys_data_dict[key])
    sys_data_dict['train_comb'] = train_comb
    
    if cell == 'nmc_temp':
        train_2_pi = System_data_list()
        train_2_pi.append(sys_data_dict['train_2_6'])
        train_2_pi.append(sys_data_dict['train_5_6'])
        sys_data_dict['train_2_pi'] = train_2_pi

    return datasets, sys_data_dict


def error(y, yhat):
    RMS = np.mean((y - yhat) ** 2) ** 0.5
    MAE = np.mean(np.abs(y - yhat))
    y_std = np.std(y, axis=0)
    NRMS = np.where(y_std != 0, RMS / y_std, np.nan)
    return RMS, NRMS, MAE


def print_error_and_plot(sys, train, *tests, plot=True, print_error=True, return_error=False, stack_vertical=False,
                         line_width=1.0, fig_name=None, dpi=300, cell=None, pbid=False, hide_train=False,
                         error_only=False, plot_error=False, plot_abs_error=False, return_sim_data=False,
                         precomputed_train_sim=None, precomputed_test_sims=None,
                         precomputed_pbid_test_sims=None):
    
    # --- Flatten tests if a single iterable (e.g. list/tuple) was passed ---
    if len(tests) == 1 and isinstance(tests[0], (list, tuple)):
        tests = tuple(tests[0])

    if error_only and not (plot_error or plot_abs_error):
        plot_abs_error = True
    
    k0 = getattr(sys, "k0", 0)
    if hide_train: 
        n_sets = len([t for t in tests if t is not None])  # valid tests
    else:
        n_sets = 1 + len([t for t in tests if t is not None])  # train + valid tests
    errors = np.zeros((n_sets, 3))  # store (RMS, NRMS, MAE) per dataset
    # legacy placeholder preserved for backward compatibility

    # Manual overrides for error-plot y-limits per cell type (set to None for auto scaling)
    error_plot_ylims = {
        "nmc_temp": (-0.15, 0.1),
        "nmc": (-0.15, 0.1),
        "lfp": (-0.04, None)}

    configured_error_ylim = None
    if cell is not None:
        configured_error_ylim = error_plot_ylims.get(cell)

    # --- Run simulations (optionally reuse precomputed results) ---
    if not hide_train:
        if precomputed_train_sim is not None:
            train_sim = precomputed_train_sim
        else:
            train_sim = sys.apply_experiment(train)
    else:
        train_sim = None

    if precomputed_test_sims is not None and len(precomputed_test_sims) != len(tests):
        raise ValueError(
            f"precomputed_test_sims length ({len(precomputed_test_sims)}) does not match number of tests ({len(tests)})"
        )

    test_sims = []
    for idx, t in enumerate(tests):
        if t is None:
            test_sims.append(None)
            continue

        if precomputed_test_sims is not None and precomputed_test_sims[idx] is not None:
            test_sims.append(precomputed_test_sims[idx])
        else:
            test_sims.append(sys.apply_experiment(t))

    if precomputed_pbid_test_sims is not None and len(precomputed_pbid_test_sims) != len(tests):
        raise ValueError(
            f"precomputed_pbid_test_sims length ({len(precomputed_pbid_test_sims)}) does not match number of tests ({len(tests)})"
        )

    # --- Collect datasets and labels ---
    pbid_test_sims = []
    pbid_initial_values = []
    for i, t in enumerate(tests):
        if t is None:
            pbid_test_sims.append(None)
            pbid_initial_values.append(0)
            continue

        if cell == 'nmc_temp':
            default_initial_values = 3
        elif cell == 'lfp':
            default_initial_values = 4
        else:
            default_initial_values = 3

        if not pbid:
            pbid_test_sims.append(None)
            pbid_initial_values.append(default_initial_values)
            continue

        precomp = None
        if precomputed_pbid_test_sims is not None:
            precomp = precomputed_pbid_test_sims[i]

        if precomp is not None:
            pbid_test_sims.append(precomp)
            pbid_initial_values.append(default_initial_values)
            continue

        if cell == 'nmc_temp':
            pbid_model_name = 'results/nmc_with_temperature/model_n,l=3,3.npy'
            kinds = ['cold', 'normal', 'hot']
            kind = kinds[i] if i < len(kinds) else kinds[-1]
            dataset = helper.load_npy_data(f'data/nmc_with_temperature/validation_data_{kind}.npy')
            pbid_model = load_model_from_file(pbid_model_name)
            initial_soc = invert_voltage_function(pbid_model.emf_function, dataset['voltage_values'][0], dataset['temperature_values'][0])
            dataset['initial_soc'] = initial_soc
            current_profile = {
                'initial_soc': initial_soc,
                'time_values': dataset['time_values'],
                'current_values': dataset['current_values'],
                'temperature_values': dataset['temperature_values'],
                'voltage_values': dataset['voltage_values'][:default_initial_values],
            }
        else:
            if cell == 'nmc':
                pbid_model_name = 'results/nmc/model_n,l=3,4.npy'
                dataset = helper.load_npy_data('data/nmc/validation_data.npy')
            elif cell == 'lfp':
                pbid_model_name = 'results/lfp/model_one_n,l=4,1.npy'
                dataset = helper.load_npy_data('data/lfp/validation_data.npy')
            else:
                raise ValueError(f"Unsupported cell type for PBID overlay: {cell}")

            pbid_model = load_model_from_file(pbid_model_name)
            current_profile = {
                'initial_soc': dataset['initial_soc'],
                'time_values': dataset['time'],
                'current_values': dataset['current'],
                'voltage_values': dataset['voltage'][:default_initial_values],
            }
        
        sim_result = simulate_model(pbid_model, current_profile)
        pbid_test_sims.append(sim_result)
        pbid_initial_values.append(default_initial_values)
            
    if cell == 'nmc_temp' and n_sets >=3:
        test_labels = ['Cold', 'Normal', 'Hot']
    elif cell != 'nmc_temp' and n_sets <= 2:
        test_labels = ['', '', '']
    else: 
        test_labels = ['1', '2', '3']

    test_entries = []
    for i, (t, s, pbid_sim, init_vals) in enumerate(zip(tests, test_sims, pbid_test_sims, pbid_initial_values)):
        if t is None:
            continue
        label = test_labels[i] if i < len(test_labels) else str(i + 1)
        test_entries.append((f"Test {label}", t, s, pbid_sim, init_vals))
        
    if hide_train:
        datasets = test_entries
    else:
        datasets = [("Train", train, train_sim, None, 0)] + test_entries


    adjusted_results = []

    # --- Compute errors ---
    for idx, (label, dset, sim, pbid_y_pred, pbid_init_vals) in enumerate(datasets):
        y_true = np.array(dset.y, copy=False)
        y_pred = np.array(sim.y, copy=False)
        pbid_y = np.array(pbid_y_pred, copy=False) if pbid_y_pred is not None else None
        
        # If extra EMF values were stored, add them back (only to voltage column if multi-output)
        dset_extra = getattr(dset, "extra", None)
        emf_values = None
        if isinstance(dset_extra, dict):
            emf_values = dset_extra.get("emf_values")

        if emf_values is not None:
            emf_values = np.asarray(emf_values)
            if y_true.ndim == 1:
                y_true = y_true + emf_values
                y_pred = y_pred + emf_values
            else:
                y_true = y_true.copy()
                y_pred = y_pred.copy()
                y_true[:, 0] = y_true[:, 0] + emf_values
                y_pred[:, 0] = y_pred[:, 0] + emf_values

        adjusted_results.append((label, y_true, y_pred, pbid_y))

        rms, nrms, mae = error(y_true[k0:], y_pred[k0:])
        errors[idx, :] = [rms, nrms, mae]

        if print_error:
            print(
                f"{label} errors (RMSE, NRMS, MAE) = "
                f"({rms*1e3:.3f} mV, {nrms:.3f}, {mae*1e3:.3f} mV)"
            )
        if pbid_y is not None:
            trim = pbid_init_vals if pbid_init_vals is not None else 0
            rms, nrms, mae = error(y_true[trim:], pbid_y[trim:])

            if print_error:
                print(
                    f"PyBatteryID {label} errors (RMSE, NRMS, MAE) = "
                    f"({rms*1e3:.3f} mV, {nrms:.3f}, {mae*1e3:.3f} mV)"
                )

    
    if cell == 'lfp':
        for idx, (label, y_true_adj, y_pred_adj, pbid_y_adj) in enumerate(adjusted_results):
            if label == 'Test ':
                y_true = y_true_adj
                y_pred = y_pred_adj
                def find_rmse_mae_for_subregions(region_slices):
                    #
                    simulated_voltage_regions = np.concatenate([ y_pred[ idx[0]:idx[1] ] for idx in region_slices ])
                    measured_voltage_regions = np.concatenate([ y_true[ idx[0]:idx[1] ] for idx in region_slices ])
                    #
                    rmse = root_mean_squared_error( simulated_voltage_regions, measured_voltage_regions )
                    mae = mean_absolute_error( simulated_voltage_regions, measured_voltage_regions )
                
                    return rmse, mae
            
                # We define slices for region A and region B
                region_A_slices = [(k0, 3350), (4600, 11330), (12570, 20820), (22100, 29320), (30600, len(y_pred))]
                #region_As_total = [(dset['time'][ idx[0]:idx[1] ] / 3600, y_pred[ idx[0]:idx[1] ]) for idx in region_A_slices]
                #
                region_B_slices = [(3350, 4600), (11330, 12570), (20820, 22100), (29320, 30600)]
                #region_Bs_total = [(dset['time'][ idx[0]:idx[1] ] / 3600, y_pred[ idx[0]:idx[1] ]) for idx in region_B_slices]
                
                # We find error values separately for regions A and B
                rmse_A, mae_A = find_rmse_mae_for_subregions(region_A_slices)
                rmse_B, mae_B = find_rmse_mae_for_subregions(region_B_slices)
                
                print(f"Region A -- red: (RMSE, MAE) = ({rmse_A * 1e3} mV, {mae_A * 1e3} mV)")
                print(f"Region B -- green: (RMSE, MAE) = ({rmse_B * 1e3} mV, {mae_B * 1e3} mV)")
                
                if pbid_y_adj is not None:
                    y_pred = pbid_y_adj
                    
                    # We find error values separately for regions A and B
                    rmse_A, mae_A = find_rmse_mae_for_subregions(region_A_slices)
                    rmse_B, mae_B = find_rmse_mae_for_subregions(region_B_slices)
                    
                    print(f"PyBatteryID Region A -- red: (RMSE, MAE) = ({rmse_A * 1e3} mV, {mae_A * 1e3} mV)")
                    print(f"PyBatteryID Region B -- green: (RMSE, MAE) = ({rmse_B * 1e3} mV, {mae_B * 1e3} mV)")
        
    if plot:    
        lpv_arx_label = "LPV-ARX"
        lpv_subnet_label = "LPV-SUBNET"
        if error_only==True:
            fig = plt.figure(figsize=(7, 2.5 * n_sets), constrained_layout=True)
            gs = fig.add_gridspec(n_sets, 1, hspace=0, wspace=0)
            axes = gs.subplots(sharex=False, sharey=True)   
        elif stack_vertical:
            fig = plt.figure(figsize=(12, 3 * n_sets), constrained_layout=True)
            gs = fig.add_gridspec(n_sets, 1, hspace=0.2, wspace=0)
            axes = gs.subplots(sharex=False, sharey=True)   
        else:
            fig = plt.figure(figsize=(5 * n_sets, 3), constrained_layout=True)
            gs = fig.add_gridspec(1, n_sets, hspace=0, wspace=0.05)
            axes = gs.subplots(sharex=False, sharey=True)
            
    
        # Ensure axes is iterable even if only 1 plot
        if n_sets == 1:
            axes = [axes]
        
        def plot_with_error(ax, y_true, y_pred, title, global_voltage_ylim, global_error_ylim,
                            plot_error=False, plot_abs_error=False, is_last=False, is_first=False,
                            pbid_y_pred=None, total_sets=1):
            x_vals = np.arange(len(y_true)) / 3600.0
            handles, labels = [], []
            error = y_true - y_pred
            abs_error = np.abs(error)
            pbid_error = None
            pbid_abs_error = None
            if pbid_y_pred is not None:
                pbid_error = y_true - pbid_y_pred
                pbid_abs_error = np.abs(pbid_error)

            if not error_only:
                # --- Voltage plots ---
                l1 = ax.plot(x_vals, y_true, "black", label="Measured", linewidth=line_width)
                l2 = ax.plot(x_vals, y_pred, label="Simulated", linewidth=line_width)
                handles += l1 + l2
                labels += ["Measured", "Simulated"]
        
                if pbid_y_pred is not None:
                    l3 = ax.plot(x_vals, pbid_y_pred, label=f"{lpv_arx_label} Simulated", linewidth=line_width)
                    handles += l3
                    labels += [f"{lpv_arx_label} Simulated"]
        
                ax.set_ylabel("Battery Voltage [V]")
                if global_voltage_ylim is not None:
                    ax.set_ylim(global_voltage_ylim)
                    
                ax.grid(visible=True, axis="both", linestyle='--')
        
                # --- Optional: overlay error/abs-error ---
                if plot_error or plot_abs_error:
                    ax2 = ax.twinx()

                    if plot_error:
                        err_line = ax2.plot(x_vals, error, label="Error", linewidth=line_width)
                        handles += err_line
                        labels += ["Error"]
                        if pbid_error is not None:
                            pbid_err_line = ax2.plot(x_vals, pbid_error, label=f"{lpv_arx_label} Error", linewidth=line_width)
                            handles += pbid_err_line
                            labels += [f"{lpv_arx_label} Error"]

                    if plot_abs_error:
                        abs_line = ax2.plot(x_vals, abs_error, label="Abs Error", linewidth=line_width)
                        handles += abs_line
                        labels += ["Abs Error"]
                        if pbid_abs_error is not None:
                            pbid_abs_line = ax2.plot(x_vals, pbid_abs_error, label=f"{lpv_arx_label} Abs Error", linewidth=line_width)
                            handles += pbid_abs_line
                            labels += [f"{lpv_arx_label} Abs Error"]
        
                    if plot_error and plot_abs_error:
                        ax2.set_ylabel("Error / Abs Error [V]")
                    elif plot_error:
                        ax2.set_ylabel("Error [V]")
                    else:
                        ax2.set_ylabel("Abs Error [V]")
        
                    if global_error_ylim is not None:
                        ax2.set_ylim(global_error_ylim)
                    if not is_last and not stack_vertical:
                        ax2.set_yticklabels([])
            else:
                # --- Error-only mode ---
                data_plotted = False

                if plot_error:
                    err_line = ax.plot(x_vals, error, label=f"{lpv_subnet_label} Error", linewidth=line_width)
                    handles += err_line
                    labels += [f"{lpv_subnet_label} Error"]
                    data_plotted = True
                    if pbid_error is not None:
                        pbid_err_line = ax.plot(x_vals, pbid_error, label=f"{lpv_arx_label} Error", linewidth=line_width)
                        handles += pbid_err_line
                        labels += [f"{lpv_arx_label} Error"]

                if plot_abs_error:
                    abs_line = ax.plot(x_vals, abs_error, label=f"{lpv_subnet_label} Abs Error", linewidth=line_width)
                    handles += abs_line
                    labels += [f"{lpv_subnet_label} Abs Error"]
                    data_plotted = True
                    if pbid_abs_error is not None:
                        pbid_abs_line = ax.plot(x_vals, pbid_abs_error, label=f"{lpv_arx_label} Abs Error", linewidth=line_width)
                        handles += pbid_abs_line
                        labels += [f"{lpv_arx_label} Abs Error"]

                if not data_plotted:
                    # Fallback to abs error to avoid empty axes
                    fallback_line = ax.plot(x_vals, abs_error, label=f"{lpv_subnet_label} Abs Error", linewidth=line_width)
                    handles += fallback_line
                    labels += [f"{lpv_subnet_label} Abs Error"]
                    if pbid_abs_error is not None:
                        pbid_abs_line = ax.plot(x_vals, pbid_abs_error, label=f"{lpv_arx_label} Abs Error", linewidth=line_width)
                        handles += pbid_abs_line
                        labels += [f"{lpv_arx_label} Abs Error"]
                    plot_abs_error = True
        
                if plot_error and plot_abs_error:
                    ax.set_ylabel("Error / Abs Error [V]")
                elif plot_error:
                    ax.set_ylabel("Error [V]")
                else:
                    ax.set_ylabel("Abs Error [V]")
        
                if global_error_ylim is not None:
                    ax.set_ylim(global_error_ylim)
                elif not plot_error:
                    ax.set_ylim(0, None)
                ax.grid(visible=True, axis="both", linestyle='--')
        
            #ax.set_xlim(0, None)
            if total_sets > 1:
                ax.set_title(title)
            if handles:
                if plot_abs_error and not plot_error:
                    legend_loc = "upper left"
                elif plot_error and not plot_abs_error:
                    legend_loc = "lower left"
                else:
                    legend_loc = "center left"
                ax.legend(handles, labels, loc=legend_loc)
                
        # --- Before looping through datasets, compute global voltage + error ranges ---
        all_voltages = []
        signed_error_series = []
        abs_error_series = []
        
        for label, y_true_plot, y_pred_plot, pbid_y_pred in adjusted_results:
            all_voltages.append(y_true_plot)
            all_voltages.append(y_pred_plot)
            if pbid_y_pred is not None:
                all_voltages.append(pbid_y_pred)
        
            if plot_error:
                signed_error_series.append(y_true_plot - y_pred_plot)
                if pbid_y_pred is not None:
                    signed_error_series.append(y_true_plot - pbid_y_pred)
            if plot_abs_error:
                abs_error_series.append(np.abs(y_true_plot - y_pred_plot))
                if pbid_y_pred is not None:
                    abs_error_series.append(np.abs(y_true_plot - pbid_y_pred))
        
        # Voltage limits (top half)
        vmin = min(np.min(v) for v in all_voltages)
        vmax = max(np.max(v) for v in all_voltages)
        voltage_range = vmax - vmin
        global_voltage_ylim = (vmin - voltage_range, vmax)
        
        # Error limits (bottom half)
        combined_errors = signed_error_series + abs_error_series
        if combined_errors:
            err_max = max(np.max(series) for series in combined_errors)
            if signed_error_series:
                err_min = min(np.min(series) for series in signed_error_series)
            else:
                err_min = 0.0
            span = err_max - err_min
            if span == 0:
                span = max(abs(err_max), 1e-6)
            padding = 0.05 * span
            lower = err_min - padding if signed_error_series else max(0.0, err_min)
            upper = err_max + padding
            global_error_ylim = (lower, upper)
        else:
            global_error_ylim = None

        if configured_error_ylim is not None:
            global_error_ylim = tuple(configured_error_ylim)
        
        # Plot all datasets
        for i, (ax, (label, y_true_plot, y_pred_plot, pbid_y_pred)) in enumerate(zip(axes, adjusted_results)):
            plot_with_error(ax, y_true_plot, y_pred_plot, label, global_voltage_ylim, global_error_ylim, plot_error=plot_error,
                            plot_abs_error=plot_abs_error, is_last=(i == len(adjusted_results) - 1), is_first=(i == 0),
                            pbid_y_pred=pbid_y_pred, total_sets=n_sets,)

        # Common x-axis label
        fig.supxlabel("Time [h]")
        
        if fig_name is not None:
            # infer format from file extension
            ext = os.path.splitext(fig_name)[1].lstrip(".").lower()
            plt.savefig(fig_name, dpi=dpi, format=ext)

    outputs = []
    if return_error:
        outputs.append(errors)
    if return_sim_data:
        outputs.append((train_sim, test_sims, pbid_test_sims))

    if outputs:
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def plot_voltage_current(dataset, line_width=1.0, temp=False, fig_name=None, dpi=300):
    
    temp_test = False;
    # Allow passing a list/tuple of datasets
    if isinstance(dataset, (list, tuple)):
        datasets = dataset
        temp_test = True;
    else:
        datasets = [dataset]

    colors = ["tab:blue", "tab:orange", "tab:red"]  # cold, normal, hot
    basecolor = 'tab:blue'

    if temp_test:
        voltage = dataset[0].y
        current = dataset[0].u.T[0]
    else:
        voltage = dataset.y
        current = dataset.u.T[0]
    N = min(len(voltage), len(current))
    time = np.arange(N) / 3600.0  # hours

    # --- Create figure(s)
    if temp:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(7, 3.45), sharex=True,gridspec_kw={"height_ratios": [2.5, 1]}, constrained_layout=True)
        ax_bg = fig.add_axes(ax1.get_position(), zorder=0, frameon=False)
        ax_bg.tick_params(axis="x", length=0, labelbottom=False)
        ax_bg.tick_params(axis="y", length=0, labelleft=False)
        ax_bg.grid(visible=True, axis='both', linestyle='--')
        ax_bg.set_axisbelow(True)
    else:
        fig, ax1 = plt.subplots(figsize=(7, 2.5), constrained_layout=True)
        ax_bg = fig.add_subplot(111)

    ax2 = ax1.twinx()

    # Make patches transparent
    ax1.patch.set_visible(False)
    ax2.patch.set_visible(False)

    # Plot current (assume same for all datasets)
    l_cur, = ax2.plot(time, current, linewidth=line_width, color='tab:grey', label="Current", zorder=1)
    ax2.set_ylabel("Current [A]")
    ax2.tick_params(axis="y")

    # Plot voltages for each dataset
    voltage_lines = []
    for i, ds in enumerate(datasets):
        color = colors[i % len(colors)]
        voltage = ds.y
        N = len(voltage)
        time = np.arange(N) / 3600.0  # hours
        if temp_test:
            line, = ax1.plot(time, voltage, color=color, linewidth=line_width,
                             label=f"Voltage [{['Cold', 'Normal', 'Hot'][i]}]", zorder=3)
        else: 
            line, = ax1.plot(time, voltage, color=basecolor, linewidth=line_width, label="Voltage", zorder=3)
        voltage_lines.append(line)

    ax1.set_ylabel("Battery Voltage [V]")
    ax1.tick_params(axis="y")

    # Set order & limits
    ax_bg.set_zorder(0)
    ax2.set_zorder(1)
    ax1.set_zorder(2)
    #ax1.set_xlim(left=0)
    #ax2.set_xlim(0, None)

    fig.canvas.draw()
    ax_bg.set_xlim(ax1.get_xlim())
    ax_bg.set_ylim(ax1.get_ylim())

    # Adjust grid size if temp plot
    if temp:
        if temp_test:
            pos = ax1.get_position()
            ax_bg.set_position([pos.x0*1.78, pos.y0 * 1.19, pos.width * 0.88, pos.height * 0.655])
        else:
            pos = ax1.get_position()
            ax_bg.set_position([pos.x0*1.76, pos.y0 * 1.18, pos.width * 0.884, pos.height * 0.655])

    # Combine legends
    lines = voltage_lines + [l_cur]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="lower left")

    # Temperature subplot (if requested)
    if temp:
        for i, ds in enumerate(datasets):
            color = colors[i % len(colors)]
            temperature = ds.u.T[1]
            N = len(temperature)
            time = np.arange(N) / 3600.0
            if temp_test:
                ax3.plot(time, temperature, linewidth=line_width, color=color,
                         label=f"Temperature [{['Cold', 'Normal', 'Hot'][i]}]")
            else:
                ax3.plot(time, temperature, linewidth=line_width, color=basecolor, label="Temperature [°C]")
        ax3.set_ylabel("Temp [°C]")
        ax3.set_xlabel("Time [h]")
        ax3.grid(visible=True, axis='both', linestyle='--')
        ax3.set_axisbelow(True)
        #ax3.legend(loc="upper right")
    else:
        ax1.set_xlabel("Time [h]")
        ax_bg.grid(visible=True, axis='both', linestyle='--')

    # Save & show
    try:
        fig.tight_layout()
    except Exception:
        pass

    if fig_name is not None:
        ext = os.path.splitext(fig_name)[1].lstrip(".").lower()
        plt.savefig(fig_name, dpi=dpi, format=ext if ext else None,
                    bbox_inches="tight", pad_inches=0.02)
    plt.show()


# -----------------------------
# Checkpoint helpers
# -----------------------------
def get_recent_checkpoint_codes(limit=5, verbose=True):
    """Return the most recent unique checkpoint codes discovered in the deepSI checkpoint directory."""
    try:
        checkpoint_dir = Path(deepSI.datasets.get_work_dirs()["checkpoints"])
    except KeyError as exc:
        raise KeyError("deepSI work directories do not expose a 'checkpoints' path.") from exc

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = sorted(
        (path for path in checkpoint_dir.iterdir() if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    collected = []
    seen_codes = set()

    for path in checkpoint_files:
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue

        code = parts[-2]
        if not code or code in seen_codes:
            continue

        seen_codes.add(code)
        collected.append((code, path))

        if len(collected) >= limit:
            break

    codes = [code for code, _ in collected]

    if verbose:
        for idx, (code, path) in enumerate(collected, start=1):
            print(f"{idx}: {code}  ({path.name})")

    return codes


# -----------------------------
# Dataset inspection helpers
# -----------------------------
def dataset_contains_vop(sys_data, raw_voltage=None, tol=1e-4, amplitude_threshold=1.5,
                         plot=False, max_points=2000, verbose=True):
    """Check whether ``sys_data.y`` holds overpotential (voltage minus EMF).

    If ``raw_voltage`` is provided, it should be the measured voltage array. The mean
    absolute difference between the stored signal and the raw voltage is reported and
    used to refine the decision. When raw voltage is unavailable, a simple amplitude
    heuristic and stored EMF metadata are used instead.
    """
    if sys_data is None:
        raise ValueError("sys_data is None; please pass a valid System_data instance.")

    y = np.asarray(sys_data.y)
    voltage_signal = y[:, 0] if y.ndim > 1 else y

    extra = getattr(sys_data, "extra", None)
    has_emf = isinstance(extra, dict) and "emf_values" in extra

    is_vop = bool(has_emf)
    mean_abs_diff = None
    max_abs_voltage = float(np.max(np.abs(voltage_signal)))
    min_voltage = float(np.min(voltage_signal))
    max_voltage = float(np.max(voltage_signal))

    if raw_voltage is not None:
        raw_voltage = np.asarray(raw_voltage)
        if raw_voltage.shape != voltage_signal.shape:
            raise ValueError(
                f"Shape mismatch: raw_voltage {raw_voltage.shape} vs signal {voltage_signal.shape}"
            )
        mean_abs_diff = float(np.mean(np.abs(voltage_signal - raw_voltage)))
        is_vop = mean_abs_diff > tol
    elif not is_vop:
        # fall back to amplitude heuristic
        is_vop = max_abs_voltage < amplitude_threshold and max_voltage < amplitude_threshold + 0.1

    if verbose:
        status = "Vop (overpotential)" if is_vop else "full voltage"
        print(f"Detected signal: {status}")
        if has_emf:
            print(" - EMF values stored in sys_data.extra['emf_values']")
        print(f" - Min/Max voltage = {min_voltage:.3f} / {max_voltage:.3f} V")
        print(f" - Max |voltage| = {max_abs_voltage:.3f} V")
        if mean_abs_diff is not None:
            print(f" - Mean |signal - raw_voltage| = {mean_abs_diff:.6f}")

    if plot:
        num_points = min(len(voltage_signal), max_points)
        plt.figure(figsize=(7, 3))
        plt.plot(np.arange(num_points), voltage_signal[:num_points], linewidth=1.0)
        plt.title(f"{'Vop' if is_vop else 'Voltage'} preview ({num_points} samples)")
        plt.xlabel("Sample index")
        plt.ylabel("Voltage [V]")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    return is_vop

