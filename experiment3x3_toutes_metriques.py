import multiprocessing
import os
import random
import shutil
import time
from copy import deepcopy

import pandas as pd
from sumo_experiments import Experiment
from sumo_experiments.preset_networks import *
from sumo_experiments.strategies import *
from sumo_experiments.traci_util import *
import numpy as np
from multiprocessing import Process, cpu_count, Manager, Value, Queue
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


class DetectionVsFixedTime3x3:

    NB_THREADS = cpu_count()
    INTERVAL_METRICS = 500

    # Fixed simulation parameter
    EXP_DURATION = 3600
    TRAINING_DURATION = 3600 * 6
    TEMP_RES_FOLDER = 'temp_results'
    PLOT_FOLDER = 'plots'
    SQUARE_SIZE = 3
    LANE_LENGTH = 200
    YELLOW_TIME = 3
    MAX_SPEED = 50
    BOOLEAN_DETECTOR_LENGTH = 20
    SATURATION_DETECTOR_LENGTH = 20

    # Variable simulation parameter
    nb_exps = 100
    seeds = [i for i in range(1000)]
    loads = [i for i in range(1000, 4000)]

    # Strategies parameters
    strategies = [
        'Fixed',
        'Max_pressure',
        'SOTL',
        # 'RL1',
        # 'RL2',
        'Acolight'
    ]
    strategies_type = {
        'Fixed': 'Traditional',
        'Max_pressure': 'Traditional',
        'SOTL': 'Traditional',
        'RL1': 'RL',
        'RL2': 'RL',
        'Acolight': 'Traditional'
    }
    max_green_times = [i for i in range(10, 121)]                 # From 10 to 120
    thresholds_switch = [i for i in range(120, 600)]              # From 120 to 599
    period_time = [i for i in range(3, 31)]                       # From 3 to 30
    flow_imbalances = [i/100 for i in range(65, 81)]              # From 0.65 to 0.80

    def __init__(self):
        manager = multiprocessing.Manager()
        self.cpt = manager.Value('i', 1)

    def do_experiment(self, params, lock):

        for param in params:

            load, flow_imbalance, strategy = param
            max_green_time = np.random.choice(self.max_green_times)
            min_green_time = np.random.choice(range(max_green_time))

            # Creating flows
            strat_type = self.strategies_type[strategy]
            if strat_type == 'Traditional':
                ns_prop = flow_imbalance
                eo_prop = 1 - flow_imbalance
                flow_north = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (ns_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                flow_south = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (ns_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                flow_east = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (eo_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                flow_west = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (eo_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                repartition = np.concatenate((flow_north, flow_east, flow_south, flow_west)).reshape(-1, 1)
                load_vector = np.array([load])
            elif strat_type == 'RL':
                # Randomly picking an episode duration
                episode_length = np.random.choice(self.episode_length)
                nb_training_episodes = self.TRAINING_DURATION // episode_length
                nb_exp_episodes = self.EXP_DURATION // episode_length \
                    if abs(((self.EXP_DURATION // episode_length) * episode_length) - self.EXP_DURATION) \
                       < abs(((self.EXP_DURATION // episode_length) * episode_length + episode_length) - self.EXP_DURATION) \
                    else (self.EXP_DURATION // episode_length) + 1
                load_vector = [np.random.choice(self.loads) for _ in range(nb_training_episodes + nb_exp_episodes)]
                repartitions = []
                for _ in range(nb_training_episodes):
                    imb = np.random.choice(self.flow_imbalances)
                    ns_prop = imb
                    eo_prop = 1 - imb
                    flow_north = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (ns_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                    flow_south = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (ns_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                    flow_east = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (eo_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                    flow_west = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (eo_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                    rep = np.concatenate((flow_north, flow_east, flow_south, flow_west)).reshape(-1, 1)
                    repartitions.append(rep)
                for _ in range(nb_exp_episodes):
                    ns_prop = flow_imbalance
                    eo_prop = 1 - flow_imbalance
                    flow_north = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (ns_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                    flow_south = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (ns_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                    flow_east = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (eo_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                    flow_west = np.ones(self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1)) * (eo_prop / 2) / (self.SQUARE_SIZE * ((self.SQUARE_SIZE * 4) - 1))
                    rep = np.concatenate((flow_north, flow_east, flow_south, flow_west)).reshape(-1, 1)
                    repartitions.append(rep)
                repartition = np.array(repartitions).T[0]

            # Create experiment objects
            net = GridNetwork(self.SQUARE_SIZE, self.SQUARE_SIZE)
            #infrastructures = net.generate_infrastructures_one_entry_phases(lane_length=self.LANE_LENGTH,
                                                                             #green_time=max_green_time,
                                                                             #yellow_time=self.YELLOW_TIME,
                                                                             #max_speed=self.MAX_SPEED)

            infrastructures = net.generate_infrastructures(lane_length=self.LANE_LENGTH,
                                                            green_time=max_green_time,
                                                            yellow_time=self.YELLOW_TIME,
                                                            max_speed=self.MAX_SPEED)


            if strat_type == 'Traditional':
                flows = net.generate_flows_with_matrix(period_time=self.EXP_DURATION,
                                                       load_vector=load_vector,
                                                       coeff_matrix=repartition,
                                                       distribution='binomial')
            elif strat_type == 'RL':
                flows = net.generate_flows_with_matrix(period_time=episode_length,
                                                       load_vector=load_vector,
                                                       coeff_matrix=repartition,
                                                       distribution='binomial')
            detectors = net.generate_all_detectors(boolean_detector_length=self.BOOLEAN_DETECTOR_LENGTH,
                                                  saturation_detector_length= self.SATURATION_DETECTOR_LENGTH)

            yellow_times = {f'x{x}-y{y}': self.YELLOW_TIME for y in range(self.SQUARE_SIZE + 1) for x in range(self.SQUARE_SIZE + 1)}
            if strategy == 'Fixed':
                phase_duration = np.random.choice(self.thresholds_switch)
                phases_duration = {f'x{x}-y{y}': [phase_duration, 3, phase_duration, 3, phase_duration, 3, phase_duration, 3] for y in range(self.SQUARE_SIZE + 1) for x in range(self.SQUARE_SIZE + 1)}
                strat = FixedTimeStrategy(infrastructures, detectors, phases_duration)
                str_params = f'phase={phase_duration}'
            elif strategy == 'Max_pressure':
                periods = {f'x{x}-y{y}': max_green_time for y in range(self.SQUARE_SIZE + 1) for x in range(self.SQUARE_SIZE + 1)}
                counted_vehicles = 'all'
                yellow_times = {f'x{x}-y{y}': 3 for y in range(self.SQUARE_SIZE + 1) for x in range(self.SQUARE_SIZE + 1)}
                strat = MaxPressureStrategy(infrastructures, detectors, periods, counted_vehicles, yellow_times)
                str_params = f'period={max_green_time}'
            elif strategy == 'SOTL':
                threshold_switch = np.random.choice(self.thresholds_switch)
                threshold_force = np.random.choice([i for i in range(2, threshold_switch)])
                thresholds_switch = {f'x{x}-y{y}': threshold_switch for y in range(self.SQUARE_SIZE + 1) for x in range(self.SQUARE_SIZE + 1)}
                thresholds_force = {f'x{x}-y{y}': threshold_force for y in range(self.SQUARE_SIZE + 1) for x in range(self.SQUARE_SIZE + 1)}
                min_phases_duration = {f'x{x}-y{y}': max_green_time for y in range(self.SQUARE_SIZE + 1) for x in range(self.SQUARE_SIZE + 1)}
                strat = SOTLStrategy(infrastructures, detectors, thresholds_switch, thresholds_force, min_phases_duration, yellow_times)
                str_params = f'threshold_switch={threshold_switch}!threshold_force={threshold_force}!min_phase_duration={max_green_time}'
            elif strategy == 'Acolight':
                max_phases_duration = {f'x{x}-y{y}': [max_green_time, max_green_time, max_green_time, max_green_time] for y in range(self.SQUARE_SIZE + 1) for x in range(self.SQUARE_SIZE + 1)}
                min_phases_duration = {f'x{x}-y{y}': [3, 3, 3, 3] for y in range(self.SQUARE_SIZE + 1) for x in range(self.SQUARE_SIZE + 1)}
                yellow_times = {f'x{x}-y{y}': 3 for y in range(self.SQUARE_SIZE + 1) for x in range(self.SQUARE_SIZE + 1)}
                strat = AcolightStrategy(infrastructures, detectors, min_phases_duration, max_phases_duration, yellow_times)
                str_params = f'max_phase_duration={max_green_time}!min_phases_duration={min_green_time}'


            # Traci functions
            if strat_type == 'Traditional':
                tw = TraciWrapper(self.EXP_DURATION)
            else:
                tw = TraciWrapper((nb_training_episodes + nb_exp_episodes) * episode_length)
            tw.add_stats_function(get_speed_data)
            tw.add_stats_function(get_co2_emissions_data)
            tw.add_behavioural_function(strat.run_all_agents)

            # Run
            exp_name = f'{strategy}-{str(load)}-{str(max_green_time)}-{str(flow_imbalance)}-{str_params}'
            exp = Experiment(
                name=exp_name,
                infrastructures=infrastructures,
                flows=flows,
                detectors=detectors
            )
            data = exp.run_traci(tw.final_function, gui=True)
            if strat_type == 'RL':
                data = data[data['simulation_step'] >= episode_length * nb_training_episodes]
            data.to_csv(f'{self.TEMP_RES_FOLDER}/{exp_name}.csv')

            lock.acquire()
            print(f"Done experiment {self.cpt.value} on {self.nb_exps * len(self.strategies)}.")
            print(f"Progress -> {round((self.cpt.value / (self.nb_exps * len(self.strategies))) * 100, 3)} %.")
            self.cpt.value += 1
            lock.release()

            exp.clean_files()


    def get_mean_matrix(self):

        # Get results files names
        res_files = os.listdir(self.TEMP_RES_FOLDER)

        results = []

        print("\nComputing aggregations of results.")
        for filename in res_files:

            data = pd.read_csv(f'{self.TEMP_RES_FOLDER}/{filename}')
            key = filename.split('.csv')[0]

            method = key.split('-')[0]
            load = int(key.split('-')[1])
            max_green_time = int(key.split('-')[2])
            imbalance = float(key.split('-')[3])
            params = key.split('-')[4]

            # Compute mean travel time
            try:
                mean_tt_data = data[['mean_travel_time', 'mean_CO2_per_travel', 'exiting_vehicles']][data['exiting_vehicles'] != 0].to_numpy()
                mean_travel_time = np.average(mean_tt_data[:, 0], weights=mean_tt_data[:, 2])
                # Compute mean co2 emissions per travel
                mean_co2_emissions = np.average(mean_tt_data[:, 1], weights=mean_tt_data[:, 2])

                results.append([method, load, imbalance, max_green_time, mean_travel_time, mean_co2_emissions, params])
            except ZeroDivisionError:
                pass

        columns = ['method', 'load', 'imbalance', 'max_green_time', 'mean_travel_time', 'mean_co2_emissions_travel', 'params']
        data_results = pd.DataFrame(results, columns=columns)
        data_results.to_csv('./data_3x3.csv')


    def plot_results(self):

        if not os.path.exists(self.PLOT_FOLDER):
            os.makedirs(self.PLOT_FOLDER)

        data = pd.read_csv('./data_3x3.csv')

        # Get potential and intensity
        nb_intervals = len(range(min(self.loads), max(self.loads), self.INTERVAL_METRICS))
        results = {}
        for strategy in self.strategies:
            results[strategy] = {'potential_travel_time': 0.0,
                                 'potential_co2_emissions': 0.0,
                                 'intensity_travel_time': 0.0,
                                 'intensity_co2_emissions': 0.0,
                                 'average_travel_time': 0.0,
                                 'average_co2_emissions': 0.0,
                                 'average_travel_time_intervals': 0.0,
                                 'average_co2_emissions_intervals': 0.0,
                                 'stddev_travel_time_intervals': 0.0,
                                 'stddev_co2_emissions_intervals' : 0.0,
                                 'median_travel_time_intervals': 0.0,
                                 'median_co2_emissions_intervals': 0.0,
                                 'decile_travel_time_intervals': 0.0,
                                 'decile_co2_emissions_intervals': 0.0,
                                 'quartile_travel_time_intervals': 0.0,
                                 'quartile_co2_emissions_intervals': 0.0,
                                 'stddev_travel_time': 0.0,
                                 'stddev_co2_emissions': 0.0,
                                 'median_travel_time': 0.0,
                                 'median_co2_emissions': 0.0,
                                 'decile_travel_time': 0.0,
                                 'decile_co2_emissions': 0.0,
                                 'quartile_travel_time': 0.0,
                                 'quartile_co2_emissions': 0.0,}
            current_data = data[data['method'] == strategy]
            for i in range(min(self.loads), max(self.loads), self.INTERVAL_METRICS):
                interval_data = current_data[np.logical_and(current_data['load'] >= i, current_data['load'] < i + self.INTERVAL_METRICS)]
                results[strategy]['potential_travel_time'] += min(interval_data['mean_travel_time']) if len(interval_data['mean_travel_time']) > 0 else 0
                results[strategy]['potential_co2_emissions'] += min(interval_data['mean_co2_emissions_travel']) if len(interval_data['mean_co2_emissions_travel']) > 0 else 0
                results[strategy]['intensity_travel_time'] += np.mean([abs(j - np.mean(interval_data['mean_travel_time'])) for j in interval_data['mean_travel_time']]) if len(interval_data['mean_travel_time']) > 0 else 0
                results[strategy]['intensity_co2_emissions'] += np.mean([abs(j - np.mean(interval_data['mean_co2_emissions_travel'])) for j in interval_data['mean_co2_emissions_travel']]) if len(interval_data['mean_co2_emissions_travel']) > 0 else 0
                results[strategy]['average_travel_time_intervals'] += np.mean(interval_data['mean_travel_time']) if len(interval_data['mean_travel_time']) > 0 else 0
                results[strategy]['average_co2_emissions_intervals'] += np.mean(interval_data['mean_co2_emissions_travel']) if len(interval_data['mean_co2_emissions_travel']) > 0 else 0
                results[strategy]['stddev_travel_time_intervals'] += np.std(interval_data['mean_travel_time']) if len(interval_data['mean_travel_time']) > 0 else 0
                results[strategy]['stddev_co2_emissions_intervals'] += np.std(interval_data['mean_co2_emissions_travel']) if len(interval_data['mean_co2_emissions_travel']) > 0 else 0
                results[strategy]['median_travel_time_intervals'] += np.median(interval_data['mean_travel_time']) if len(interval_data['mean_travel_time']) > 0 else 0
                results[strategy]['median_co2_emissions_intervals'] += np.median(interval_data['mean_co2_emissions_travel']) if len(interval_data['mean_co2_emissions_travel']) > 0 else 0
                results[strategy]['decile_travel_time_intervals'] += np.percentile(interval_data['mean_travel_time'], 10) if len(interval_data['mean_travel_time']) > 0 else 0
                results[strategy]['decile_co2_emissions_intervals'] += np.percentile(interval_data['mean_co2_emissions_travel'], 10) if len(interval_data['mean_co2_emissions_travel']) > 0 else 0
                results[strategy]['quartile_travel_time_intervals'] += np.percentile(interval_data['mean_travel_time'], 25) if len(interval_data['mean_travel_time']) > 0 else 0
                results[strategy]['quartile_co2_emissions_intervals'] += np.percentile(interval_data['mean_co2_emissions_travel'], 25) if len(interval_data['mean_co2_emissions_travel']) > 0 else 0
            results[strategy]['potential_travel_time'] = results[strategy]['potential_travel_time'] / nb_intervals
            results[strategy]['potential_co2_emissions'] = results[strategy]['potential_co2_emissions'] / nb_intervals
            results[strategy]['intensity_travel_time'] = results[strategy]['intensity_travel_time'] / nb_intervals
            results[strategy]['intensity_co2_emissions'] = results[strategy]['intensity_co2_emissions'] / nb_intervals
            results[strategy]['average_travel_time_intervals'] = results[strategy]['average_travel_time_intervals'] / nb_intervals
            results[strategy]['average_co2_emissions_intervals'] = results[strategy]['average_co2_emissions_intervals'] / nb_intervals
            results[strategy]['stddev_travel_time_intervals'] = results[strategy]['stddev_travel_time_intervals'] / nb_intervals
            results[strategy]['stddev_co2_emissions_intervals'] = results[strategy]['stddev_co2_emissions_intervals'] / nb_intervals
            results[strategy]['median_travel_time_intervals'] = results[strategy]['median_travel_time_intervals'] / nb_intervals
            results[strategy]['median_co2_emissions_intervals'] = results[strategy]['median_co2_emissions_intervals'] / nb_intervals
            results[strategy]['decile_travel_time_intervals'] = results[strategy]['decile_travel_time_intervals'] / nb_intervals
            results[strategy]['decile_co2_emissions_intervals'] = results[strategy]['decile_co2_emissions_intervals'] / nb_intervals
            results[strategy]['quartile_travel_time_intervals'] = results[strategy]['quartile_travel_time_intervals'] / nb_intervals
            results[strategy]['quartile_co2_emissions_intervals'] = results[strategy]['quartile_co2_emissions_intervals'] / nb_intervals
            results[strategy]['average_travel_time'] = np.mean(current_data['mean_travel_time'])
            results[strategy]['average_co2_emissions'] = np.mean(current_data['mean_co2_emissions_travel'])
            results[strategy]['stddev_travel_time'] = np.std(current_data['mean_travel_time'])
            results[strategy]['stddev_co2_emissions'] = np.std(current_data['mean_co2_emissions_travel'])
            results[strategy]['median_travel_time'] = np.median(current_data['mean_travel_time'])
            results[strategy]['median_co2_emissions'] = np.median(current_data['mean_co2_emissions_travel'])
            results[strategy]['decile_travel_time'] = np.percentile(current_data['mean_travel_time'], 10)
            results[strategy]['decile_co2_emissions'] = np.percentile(current_data['mean_co2_emissions_travel'], 10)
            results[strategy]['quartile_travel_time'] = np.percentile(current_data['mean_travel_time'], 25)
            results[strategy]['quartile_co2_emissions'] = np.percentile(current_data['mean_co2_emissions_travel'], 25)

        final_results = []
        for strategy in self.strategies:
            potential_travel_time = round(results[strategy]['potential_travel_time'], 2)
            potential_emissions = round(results[strategy]['potential_co2_emissions'], 2)
            intensity_travel_time = round(results[strategy]['intensity_travel_time'], 2)
            intensity_emissions = round(results[strategy]['intensity_co2_emissions'], 2)
            average_travel_time_intervals = round(results[strategy]['average_travel_time_intervals'], 2)
            average_co2_emissions_intervals = round(results[strategy]['average_co2_emissions_intervals'], 2)
            stddev_travel_time_intervals = round(results[strategy]['stddev_travel_time_intervals'], 2)
            stddev_co2_emissions_intervals = round(results[strategy]['stddev_co2_emissions_intervals'], 2)
            median_travel_time_intervals = round(results[strategy]['median_travel_time_intervals'], 2)
            median_co2_emissions_intervals = round(results[strategy]['median_co2_emissions_intervals'], 2)
            decile_travel_time_intervals = round(results[strategy]['decile_travel_time_intervals'], 2)
            decile_co2_emissions_intervals = round(results[strategy]['decile_co2_emissions_intervals'], 2)
            quartile_travel_time_intervals = round(results[strategy]['quartile_travel_time_intervals'], 2)
            quartile_co2_emissions_intervals = round(results[strategy]['quartile_co2_emissions_intervals'], 2)
            average_travel_time = round(results[strategy]['average_travel_time'], 2)
            average_co2_emissions = round(results[strategy]['average_co2_emissions'], 2)
            stddev_travel_time = round(results[strategy]['stddev_travel_time'], 2)
            stddev_co2_emissions = round(results[strategy]['stddev_co2_emissions'], 2)
            median_travel_time = round(results[strategy]['median_travel_time'], 2)
            median_co2_emissions = round(results[strategy]['median_co2_emissions'], 2)
            decile_travel_time = round(results[strategy]['decile_travel_time'], 2)
            decile_co2_emissions = round(results[strategy]['decile_co2_emissions'], 2)
            quartile_travel_time = round(results[strategy]['quartile_travel_time'], 2)
            quartile_co2_emissions = round(results[strategy]['quartile_co2_emissions'], 2)
            final_results.append([strategy,
                                  potential_travel_time,
                                  potential_emissions,
                                  intensity_travel_time,
                                  intensity_emissions,
                                  average_travel_time_intervals,
                                  average_co2_emissions_intervals,
                                  stddev_travel_time_intervals,
                                  stddev_co2_emissions_intervals,
                                  median_travel_time_intervals,
                                  median_co2_emissions_intervals,
                                  decile_travel_time_intervals,
                                  decile_co2_emissions_intervals,
                                  quartile_travel_time_intervals,
                                  quartile_co2_emissions_intervals,
                                  average_travel_time,
                                  average_co2_emissions,
                                  stddev_travel_time,
                                  stddev_co2_emissions,
                                  median_travel_time,
                                  median_co2_emissions,
                                  decile_travel_time,
                                  decile_co2_emissions,
                                  quartile_travel_time,
                                  quartile_co2_emissions])
        final_results = pd.DataFrame(final_results, columns=['Strategy',
                                                             'Potential travel time',
                                                             'Potential CO2 emissions',
                                                             'Intensity travel time',
                                                             'Intensity CO2 emissions',
                                                             'average_travel_time_intervals',
                                                             'average_co2_emissions_intervals',
                                                             'stddev_travel_time_intervals',
                                                             'stddev_co2_emissions_intervals',
                                                             'median_travel_time_intervals',
                                                             'median_co2_emissions_intervals',
                                                             'decile_travel_time_intervals',
                                                             'decile_co2_emissions_intervals',
                                                             'quartile_travel_time_intervals',
                                                             'quartile_co2_emissions_intervals',
                                                             'average_travel_time',
                                                             'average_co2_emissions',
                                                             'stddev_travel_time',
                                                             'stddev_co2_emissions',
                                                             'median_travel_time',
                                                             'median_co2_emissions',
                                                             'decile_travel_time',
                                                             'decile_co2_emissions',
                                                             'quartile_travel_time',
                                                             'quartile_co2_emissions'])
        final_results.to_csv('metrics_3x3_toutes_metriques.csv', index=False)

        data['method'] = data['method'].map(lambda x: x.replace('_', ' '))
        data['mean_co2_emissions_travel'] = data['mean_co2_emissions_travel'].map(lambda x: x / 10**6)

        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        markers = ['X', '^', 'D', 'o', 'P']
        colors = ['#e41a1c', '#636363', '#4daf4a', '#393b79', '#e6ab02']
        sns.scatterplot(ax=ax1, x='load', y='mean_travel_time', data=data, style='method', hue='method', palette=colors, markers=markers, s=84)
        sns.scatterplot(ax=ax2, x='load', y='mean_co2_emissions_travel', data=data, style='method', hue='method', palette=colors, markers=markers, s=84)
        # g = sns.pairplot(data=data,
        #              hue='method',
        #              x_vars=['flow'],# 'asymetry', 'complexity'],
        #              y_vars=['mean_travel_time', 'mean_speed', 'mean_co2_emissions'])
        #
        # g.fig.suptitle("3x3 grid", y=1.08)
        fig.set_size_inches(45, 15)
        ax1.set_xlabel('Load (in vehs/h)', fontsize=30, labelpad=20)
        ax2.set_xlabel('Load (in vehs/h)', fontsize=30, labelpad=20)
        ax1.set_ylabel('Mean travel time (in s)', fontsize=30, labelpad=20)
        ax1.set_ylim(0, 2000)
        ax2.set_ylabel('CO2 emissions per travel (in Kg)', fontsize=30, labelpad=20)
        ax2.set_ylim(0, 4)
        ax1.legend(fontsize=24)
        ax2.legend(fontsize=24)
        plt.savefig(f'{self.PLOT_FOLDER}/3x3.png')

        # method_convertor = {'boolean': 1,
        #                     'fixed': 2}
        #
        # data['method'] = data['method'].map(lambda x: method_convertor[x])
        # data['flow_method'] = data['flow'] * data['method']

        #print(partial_corr(data=data, x='flow', y='mean_travel_time', covar=['method', 'max_green_time'], method='pearson'))

        # corr_matrix = abs(data[
        #                       ['flow', 'asymetry', 'complexity', 'mean_travel_time', 'mean_speed', 'mean_co2_emissions',
        #                        'method', 'flow_method']].corr())
        # fig, ax = plt.subplots(figsize=(10, 10))
        # sns.heatmap(ax=ax, data=corr_matrix[['flow', 'asymetry', 'complexity', 'method', 'flow_method']][3:-2],
        #             cmap='Blues', annot=True)
        # plt.xticks(rotation=30)
        plt.show()

    def test(self):
        results = {}
        filenames = {}
        for file in os.listdir(self.TEMP_RES_FOLDER):
            flows = file.split('-')[1]
            if flows not in results:
                results[flows] = 1
                filenames[flows] = [file]
            else:
                results[flows] += 1
                filenames[flows].append(file)
        good_keys = []
        for key in results:
            if results[key] == 5:
                good_keys.append(filenames[key])
        print(good_keys)



    def run(self):

        t = time.time()

        # Create temp folder
        if os.path.exists(self.TEMP_RES_FOLDER):
            shutil.rmtree(self.TEMP_RES_FOLDER)
        os.makedirs(self.TEMP_RES_FOLDER)

        self.cpt.value = 1
        loads = [np.random.choice(self.loads) for _ in range(self.nb_exps)] * len(self.strategies)
        imbalances = [np.random.choice(self.flow_imbalances) for _ in range(self.nb_exps)] * len(self.strategies)
        strats = list(np.array([[strat] * self.nb_exps for strat in self.strategies]).flatten())
        params = list(zip(loads, imbalances, strats))

        # Split all configurations in cpu_count() lists
        work_vectors = [[] for _ in range(self.NB_THREADS)]
        for i in range(len(params)):
            index = i % self.NB_THREADS
            work_vectors[index].append(params[i])

        # Create workers
        workers = []
        manager = Manager()
        lock = manager.Lock()
        for i in range(self.NB_THREADS):
            workers.append(Process(target=self.do_experiment, args=(work_vectors[i], lock)))

        # Start workers
        for worker in workers:
            worker.start()

        # Join workers
        for worker in workers:
            worker.join()

        self.get_mean_matrix()
        self.plot_results()
        #shutil.rmtree(self.TEMP_RES_FOLDER)

        duration = time.time() - t
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        print(f"Durée de l'expérience : {hours} heures et {minutes} minutes.")


if __name__ == "__main__":
    exp = DetectionVsFixedTime3x3()
    exp.run()
