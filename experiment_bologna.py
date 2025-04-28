import multiprocessing
import os
import random
import shutil
import time
from copy import deepcopy

import pandas as pd
from sumo_experiments import Experiment
from sumo_experiments.preset_networks import *
from sumo_experiments.strategies.bologna import *
from sumo_experiments.traci_util import *
import numpy as np
from multiprocessing import Process, cpu_count, Manager, Value, Queue
#from progress.bar import Bar
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


class BolognaExperiment:

    NB_THREADS = cpu_count()
    INTERVAL_METRICS = 0.25

    # Fixed simulation parameter
    EXP_DURATION = 4800
    TRAINING_DURATION = 3600 * 6
    TEMP_RES_FOLDER = 'temp_results'
    PLOT_FOLDER = 'plots'
    DATA_FILE = 'res_bologna.csv'
    METRICS_FILE = 'metrics_bologna.csv'
    BOOLEAN_DETECTOR_LENGTH = 20

    # Variable simulation parameter
    nb_exps = 80
    seeds = [i for i in range(1000)]
    coeffs = [i/100 for i in range(50, 150)]

    # Strategies parameters
    strategies = [
        'Fixed',
        'Max_pressure',
        'SOTL',
        'Acolight'
    ]
    max_green_times = [i for i in range(20, 121)]                 # From 10 to 120
    thresholds_switch = [i for i in range(120, 1200)]              # From 120 to 599
    thresholds_force = [i for i in range(30, 70, 5)]              # From 30 to 70
    period_time = [i for i in range(3, 31)]                       # From 3 to 30

    def __init__(self):
        manager = multiprocessing.Manager()
        self.cpt = manager.Value('i', 1)

    def do_experiment(self, params, lock):

        for param in params:

            coeff, strategy = param

            # Create experiment objects
            net = BolognaNetwork()
            net.generate_flows(coeff)

            if strategy == 'Fixed':
                phase_times = {}
                phase_times['210'] = {0: np.random.choice(self.max_green_times), 2: np.random.choice(self.max_green_times), 4: np.random.choice(self.max_green_times)}
                phase_times['219'] = {0: np.random.choice(self.max_green_times), 2: np.random.choice(self.max_green_times), 4: np.random.choice(self.max_green_times)}
                phase_times['209'] = {0: np.random.choice(self.max_green_times), 2: np.random.choice(self.max_green_times)}
                phase_times['220'] = {0: np.random.choice(self.max_green_times), 2: np.random.choice(self.max_green_times)}
                phase_times['221'] = {0: np.random.choice(self.max_green_times), 3: np.random.choice(self.max_green_times)}
                phase_times['235'] = {0: np.random.choice(self.max_green_times), 4: np.random.choice(self.max_green_times), 6: np.random.choice(self.max_green_times), 8: np.random.choice(self.max_green_times)}
                phase_times['273'] = {0: np.random.choice(self.max_green_times), 2: np.random.choice(self.max_green_times), 4: np.random.choice(self.max_green_times)}
                strat = FixedTimeStrategyBologna(phase_times)
                phase_times_label = ''
                for i in ['209', '210', '219', '220', '221', '235', '273']:
                    phase_times_label += str(phase_times[i]) + '$'
                str_params = f'phase_times={phase_times_label}'
            elif strategy == 'Max_pressure':
                periods = {}
                periods_label = ''
                for i in ['209', '210', '219', '220', '221', '235', '273']:
                    periods[i] = np.random.choice(self.max_green_times)
                    periods_label += str(periods[i]) + '$'
                strat = MaxPressureStrategyBologna(periods)
                str_params = f'periods={periods_label}'
            elif strategy == 'SOTL':
                t_switch = {}
                t_force = {}
                min_duration = {}
                t_force_label = ''
                t_switch_label = ''
                min_durations_label = ''
                for i in ['209', '210', '219', '220', '221', '235', '273']:
                    t_switch[i] = np.random.choice(self.thresholds_switch)
                    t_switch_label += str(t_switch[i]) + '$'
                    t_force[i] = np.random.choice(self.thresholds_force)
                    t_force_label += str(t_force[i]) + '$'
                    min_duration[i] = np.random.choice(self.max_green_times)
                    min_durations_label += str(min_duration[i]) + '$'
                strat = SotlStrategyBologna(t_switch, t_force, min_duration)
                str_params = f'threshold_switch={t_switch_label}!threshold_force={t_force_label}!min_phase_duration={min_durations_label}'
            elif strategy == 'Acolight':
                max_duration = {}
                min_duration = {}
                max_duration_label = ''
                min_durations_label = ''
                for i in ['209', '210', '219', '220', '221', '235', '273']:
                    max_duration[i] = np.random.choice(self.max_green_times)
                    max_duration_label += str(max_duration[i]) + '$'
                    min_duration[i] = 3
                    min_durations_label += str(min_duration[i]) + '$'
                strat = AcolightStrategyBologna(min_duration, max_duration)
                str_params = f'min_phases_duration={min_durations_label}!max_phases_duration={max_duration_label}'

            # Traci functions
            tw = TraciWrapper(self.EXP_DURATION)
            tw.add_stats_function(get_speed_data)
            tw.add_stats_function(get_co2_emissions_data)
            tw.add_behavioural_function(strat.run_all_agents)

            # Run
            exp_name = f'{strategy}-{str(coeff)}-{str_params}'
            exp = Experiment(
                name=exp_name,
                full_line_command=net.FULL_LINE_COMMAND
            )

            try:
                data = exp.run_traci(tw.final_function, no_warnings=True, gui=True)#nb_threads=8)
                data.to_csv(f'{self.TEMP_RES_FOLDER}/{exp_name}.csv')

                print(f"Done experiment {self.cpt.value} on {self.nb_exps * len(self.strategies)}.")
                self.cpt.value += 1
            except Exception as e:
                print(e)

            exp.clean_files()
            net.clean_files()


    def get_mean_matrix(self):

        res_files = os.listdir(self.TEMP_RES_FOLDER)

        results = []

        print("\nComputing aggregations of results.")
        for filename in res_files:

            data = pd.read_csv(f'{self.TEMP_RES_FOLDER}/{filename}')
            key = filename.split('.csv')[0]

            method = key.split('-')[0]
            load = float(key.split('-')[1])
            params = key.split('-')[2]

            # Compute mean travel time
            try:
                mean_tt_data = data[['mean_travel_time', 'mean_CO2_per_travel', 'exiting_vehicles']][data['exiting_vehicles'] != 0].to_numpy()
                mean_travel_time = np.average(mean_tt_data[:, 0], weights=mean_tt_data[:, 2])
                # Compute mean co2 emissions per travel
                mean_co2_emissions = np.average(mean_tt_data[:, 1], weights=mean_tt_data[:, 2])

                results.append([method, load, mean_travel_time, mean_co2_emissions, params])
            except ZeroDivisionError:
                pass

        columns = ['method', 'coeff', 'mean_travel_time', 'mean_co2_emissions_travel', 'params']
        data_results = pd.DataFrame(results, columns=columns)
        data_results.to_csv(self.DATA_FILE)


    def plot_results(self):

        if not os.path.exists(self.PLOT_FOLDER):
            os.makedirs(self.PLOT_FOLDER)

        data = pd.read_csv(self.DATA_FILE)

        # Get potential and intensity
        nb_intervals = len(range(int(min(self.coeffs) * 100), int(max(self.coeffs) * 100), int(self.INTERVAL_METRICS * 100)))
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
            for i in range(int(min(self.coeffs) * 100), int(max(self.coeffs) * 100), int(self.INTERVAL_METRICS * 100)):
                coeff = i / 100
                interval_data = current_data[np.logical_and(current_data['coeff'] >= coeff, current_data['coeff'] < coeff + self.INTERVAL_METRICS)]
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
        final_results.to_csv('all_metrics_bologna.csv', index=False)

        data['method'] = data['method'].map(lambda x: x.replace('_', ' '))
        data['mean_co2_emissions_travel'] = data['mean_co2_emissions_travel'].map(lambda x: x / 10**6)

        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        markers = ['X', '^', 'D', 'o', 'P']
        colors = ['#e41a1c', '#636363', '#4daf4a', '#393b79', '#e6ab02']
        sns.scatterplot(ax=ax1, x='coeff', y='mean_travel_time', data=data, style='method', hue='method', palette=colors, markers=markers, s=84)
        sns.scatterplot(ax=ax2, x='coeff', y='mean_co2_emissions_travel', data=data, style='method', hue='method', palette=colors, markers=markers, s=84)
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
        ax1.set_ylim(0, 1250)
        ax2.set_ylabel('CO2 emissions per travel (in Kg)', fontsize=30, labelpad=20)
        ax2.set_ylim(0, 1.5)
        ax1.legend(fontsize=24)
        ax2.legend(fontsize=24)
        plt.savefig(f'{self.PLOT_FOLDER}/bologna.png')

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
        coeffs = [np.random.choice(self.coeffs) for _ in range(self.nb_exps)] * len(self.strategies)
        strats = list(np.array([[strat] * self.nb_exps for strat in self.strategies]).flatten())
        params = list(zip(coeffs, strats))

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
    exp = BolognaExperiment()
    exp.plot_results()
