from typing import List, Tuple, Dict, Iterable
from pygsti.circuits import Circuit
import pygsti
import matplotlib.pyplot as plt

from quapack.pyRPE import RobustPhaseEstimation
from quapack.pyRPE.quantum import Q as _Qrpe
from abc import ABC, abstractmethod
import numpy as np


def estimate_phase(depths, cos_plus_counts, cos_minus_counts, sin_plus_counts, sin_minus_counts):
    """
    Estimate the phase of germ in a dataset using RPE.
    """   
    experiment = _Qrpe()
    for idx, d in enumerate(depths):
        experiment.process_cos(d, (int(cos_plus_counts[idx]), int(cos_minus_counts[idx])))
        experiment.process_sin(d, (int(sin_plus_counts[idx]), int(sin_minus_counts[idx])))
    analysis = RobustPhaseEstimation(experiment)
    phase_estimates = analysis.angle_estimates
    last_good_idx = analysis.check_unif_local(historical=True)
    return phase_estimates, last_good_idx

def plot_outcome_dist(dist, num_qubits, ax=None, color='blue'):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(12, 6))
    # genetate binary strings num_qubits long
    binary_strings = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]
    # get the counts in the order of the binary strings
    counts = []
    for b in binary_strings:
        if b not in dist:
            counts.append(0)
        else:
            counts.append(dist[b])
    # plot the counts
    ax.bar(binary_strings, counts, alpha=0.5, color=color)
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Counts')

def plot_signal_on_circle(signal, depths, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(12, 6))
    # plot the signals on the complex plane with a colormap for the depth
    for idx, d in enumerate(depths):
        ax.scatter(signal[idx].real, signal[idx].imag, color=plt.cm.viridis(idx/(len(depths)-1)))
    ax.set_title(title)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_aspect('equal')
    ax.grid()
    # add colorbar 
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(depths)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Depth index')
    # draw the unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_artist(circle)
    # set the axis limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)


class AnalysisBase(ABC):
    def __init__(self, dataset, edesign):
        self.dataset = dataset
        self.edesign = edesign
        self.rpe_outcome_dict = self._construct_rpe_outcome_dict()
        self.raw_estimates, self.last_good_idxs = self._make_raw_estimates()
        self.signals = self.extract_signals(dataset)

    def __str__(self) -> str:
        return f"Estimates: {self.phase_estimates},\n Last Good Index: {self.last_good_idx}"
    
    @property
    def estimates(self):
        raise NotImplementedError
    
    @property
    def best_raw(self):
        raise NotImplementedError
    
    def _construct_rpe_outcome_dict(self):
        rpe_count_dict = {}
        for germ in self.edesign.circuit_dict.keys():
            rpe_count_dict[germ] = {}
            for measurement_label in self.edesign.circuit_dict[germ].keys():
                rpe_count_dict[germ][measurement_label] = {}
                for typ in ['I', 'Q']:
                    rpe_count_dict[germ][measurement_label][typ] = {
                        '+': [],
                        '-': []
                    }
                    for depth_idx, depth in enumerate(self.edesign.depths):
                        circ = self.edesign.circuit_dict[germ][measurement_label][typ][depth]
                        counts = self.dataset[circ].counts
                        plus_count_labels = self.edesign.signal_subspaces[germ][measurement_label][typ]['+']
                        minus_count_labels = self.edesign.signal_subspaces[germ][measurement_label][typ]['-']
                        plus_counts = sum([counts[label] for label in plus_count_labels])
                        minus_counts = sum([counts[label] for label in minus_count_labels])
                        rpe_count_dict[germ][measurement_label][typ]['+'].append(plus_counts)
                        rpe_count_dict[germ][measurement_label][typ]['-'].append(minus_counts)
        return rpe_count_dict
    
    def _make_raw_estimates(self):
        phase_estimates = {}
        last_good_idx = {}
        for idx, germ in enumerate(self.edesign.circuit_dict.keys()):
            phase_estimates[germ] = {}
            last_good_idx[germ] = {}
            for measurement_label in self.edesign.circuit_dict[germ].keys():
                germ_phase, germ_lgg = estimate_phase(self.edesign.depths, 
                                                    self.rpe_outcome_dict[germ][measurement_label]['I']['+'],
                                                    self.rpe_outcome_dict[germ][measurement_label]['I']['-'], 
                                                    self.rpe_outcome_dict[germ][measurement_label]['Q']['+'], 
                                                    self.rpe_outcome_dict[germ][measurement_label]['Q']['-'])
                phase_estimates[germ][measurement_label] = germ_phase
                last_good_idx[germ][measurement_label] = germ_lgg
        return phase_estimates, last_good_idx


    
    # def _make_estimate(self):
    #     phase_estimates = {}
    #     last_good_idx = {}
    #     for idx, germ in enumerate(self.edesign.germs):
    #         germ_phase, germ_lgg = estimate_phase(self.edesign.depths, 
    #                                                         self.rpe_outcome_dict[tuple(germ)]['I']['+'],
    #                                                         self.rpe_outcome_dict[tuple(germ)]['I']['-'], 
    #                                                         self.rpe_outcome_dict[tuple(germ)]['Q']['+'], 
    #                                                         self.rpe_outcome_dict[tuple(germ)]['Q']['-'])
            
    #         phase_estimates[tuple(germ)] = germ_phase
    #         last_good_idx[tuple(germ)] = germ_lgg
    #     return phase_estimates, last_good_idx

            
    def plot_dataset(self, target_model=None):
        """Make a bunch of subplots for the dataset."""
        ds = self.dataset
        for germ in self.edesign.preparation_fiducials.keys():
            for meas in self.edesign.preparation_fiducials[germ].keys():
                fig, ax = plt.subplots(2, len(self.edesign.depths), figsize=(len(self.edesign.depths)*3, 10), sharey=True)
                # set the germ as the super title
                fig.suptitle(str(germ) + ' ' + str(meas))
                for idx, depth in enumerate(self.edesign.depths):
                    inphase_circ  = self.edesign.circuit_dict[germ][meas]['I'][depth]
                    quad_circ = self.edesign.circuit_dict[germ][meas]['Q'][depth]
                    inphase_data = ds[inphase_circ].counts
                    quad_data = ds[quad_circ].counts
                    

                    plot_outcome_dist(inphase_data, num_qubits=len(self.edesign.qids), ax=ax[0, idx])
                    plot_outcome_dist(quad_data, num_qubits=len(self.edesign.qids), ax=ax[1, idx])
                    # add title that is the depth
                    ax[0, idx].set_title(f'I at {depth}')
                    ax[1, idx].set_title(f'Q at {depth}')

                    if target_model is not None:
                        inphase_target_probs = target_model.probabilities(inphase_circ)
                        num_counts = sum(inphase_data.values())
                        expected_outcomes = { key[0]: num_counts*val for key, val in inphase_target_probs.items()}
                        plot_outcome_dist(expected_outcomes, num_qubits=len(self.edesign.qids), ax=ax[0, idx], color='red')
                        num_counts = sum(quad_data.values())
                        quad_target_probs = target_model.probabilities(quad_circ)
                        expected_outcomes = { key[0]: num_counts*val for key, val in quad_target_probs.items()}
                        plot_outcome_dist(expected_outcomes, num_qubits=len(self.edesign.qids), ax=ax[1, idx], color='red')

                # add legend 
                ax[0, 0].legend(['Data', 'Target'])


                    # ax[0, idx].bar([0, 1], [inphase_data['0'], inphase_data['1']])
                    # if target_model is not None:
                    #     ax[0, idx].bar([0, 1], [inphase_target_probs['0']*num_counts, inphase_target_probs['1']*num_counts], alpha=0.5)
                    # ax[0, idx].set_title(f'I at {depth}')
                    # ax[1, idx].bar([0, 1], [quad_data['0'], quad_data['1']])
                    # if target_model is not None:
                    #     ax[1, idx].bar([0, 1], [quad_target_probs['0']*num_counts, quad_target_probs['1']*num_counts], alpha=0.5)
                    # ax[1, idx].set_title(f'Q at {depth}')
                # set share axis
                plt.tight_layout()

    def extract_signals(self, dataset):
        signals = {}
        for germ in self.rpe_outcome_dict.keys():
            signals[germ] = {}
            for measurement in self.rpe_outcome_dict[germ].keys():
                signals[germ][measurement] = []
                inphase_counts = self.rpe_outcome_dict[germ][measurement]['I']
                quadrature_counts = self.rpe_outcome_dict[germ][measurement]['Q']
                for idx, d in enumerate(self.edesign.depths):
                    inphase_plus = inphase_counts['+'][idx]
                    inphase_minus = inphase_counts['-'][idx]
                    quadrature_plus = quadrature_counts['+'][idx]
                    quadrature_minus = quadrature_counts['-'][idx]
                    try:
                        s_real = 1 - 2 * inphase_plus/(inphase_plus + inphase_minus)
                        s_imag = 1 - 2 * quadrature_plus/(quadrature_plus + quadrature_minus)
                    except:
                        s_real = 0
                        s_imag = 0
                    signals[germ][measurement].append(s_real + 1j*s_imag)
        return signals
        # for param_label in self.rpe_outcome_dict.keys():
        #     signals[param_label] = []
        #     inphase_counts = self.rpe_outcome_dict[param_label]['I']
        #     quadrature_counts = self.rpe_outcome_dict[param_label]['Q']
        #     for idx, d in enumerate(self.edesign.depths):
        #         inphase_plus = inphase_counts['+'][idx]
        #         inphase_minus = inphase_counts['-'][idx]
        #         quadrature_plus = quadrature_counts['+'][idx]
        #         quadrature_minus = quadrature_counts['-'][idx]
        #         try:
        #             s_real = 1 - 2 * inphase_plus/(inphase_plus + inphase_minus)
        #             s_imag = 1 - 2 * quadrature_plus/(quadrature_plus + quadrature_minus)
        #         except:
        #             s_real = 0
        #             s_imag = 0
        #         signals[param_label].append(s_real + 1j*s_imag)
        # return signals
    
    
    def plot_all_signals(self):
        for germ in self.signals.keys():
            for measurment in self.signals[germ].keys():
                fig, ax = plt.subplots(1, figsize=(12, 6))
                plot_signal_on_circle(self.signals[germ][measurment], self.edesign.depths, ax=ax, title=str(germ) + ' ' + measurment)
                plt.show()

    
        
class Analysis_XI(AnalysisBase):
    def __init__(self, dataset, edesign):
        super().__init__(dataset, edesign)

    @property
    def estimates(self):
        best_raw = self.best_raw
        estimate_axis = -np.sin(best_raw['ZXXZZXXZ']/2)/(2*np.cos(np.pi*best_raw['X']/2))
        return {
            'X overrot' : -best_raw['X'] - np.pi/2, 
            'X axis' : -estimate_axis, 
            'idle' : -best_raw['I']
        }

        pass
        # raw0 = self.phase_estimates[tuple(self.edesign.germs[0])][self.last_good_idxs[tuple(self.edesign.germs[0])]]
        # raw1 = self.phase_estimates[tuple(self.edesign.germs[1])][self.last_good_idxs[tuple(self.edesign.germs[1])]]
        # raw2 = self.phase_estimates[tuple(self.edesign.germs[2])][self.last_good_idxs[tuple(self.edesign.germs[2])]]
        # raw0_rectified = (raw0 + np.pi) % (2*np.pi) - np.pi
        # estimate0_unrectified = -raw0_rectified/(np.pi/2) - 1
        # estimate0 = (estimate0_unrectified + np.pi) % (2*np.pi) - np.pi
        # raw1_rectified = (raw1 + np.pi) % (2*np.pi) - np.pi
        # estimate1 = -np.sin(raw1_rectified/2)/(2*np.cos(np.pi*estimate0/2))
        # estimate2 = -(raw2 + np.pi) % (2*np.pi) - np.pi
        # return {
        #     tuple(self.edesign.germs[0]): estimate0,
        #     tuple(self.edesign.germs[1]): estimate1,
        #     tuple(self.edesign.germs[2]): estimate2
        # }
    
    @property
    def best_raw(self):
        germ0 = self.edesign.germs[0]
        germ1 = self.edesign.germs[1]
        germ2 = self.edesign.germs[2]
        raws = {
            'X' : self.raw_estimates[germ0]['0'][self.last_good_idxs[germ0]['0']],
            'ZXXZZXXZ' : self.raw_estimates[germ1]['0'][self.last_good_idxs[germ1]['0']],
            'I' : self.raw_estimates[germ2]['+'][self.last_good_idxs[germ2]['+']],
        }
        return {key : (raws[key] + np.pi) % (2*np.pi) - np.pi  for key in raws.keys()}

    

class Analysis_CZ(AnalysisBase):
    def __init__(self, dataset, edesign):
        super().__init__(dataset, edesign)

    @property
    def estimates(self):
        model_to_raw = np.array([
            [1, 0, 1],
            [1, 0, -1],
            [0, 1, -1]
        ])
        raw_estimates = np.array([
            self.best_raw['0+'],
            self.best_raw['1+'],
            self.best_raw['+1']
        ])
        raw_estimates[0] = (raw_estimates[0] + np.pi)% (2*np.pi) - np.pi
        raw_estimates[1:] = raw_estimates[1:] % (2*np.pi) 
        estimates = np.linalg.solve(model_to_raw, raw_estimates)
        estimates = (estimates + np.pi) % (2*np.pi) - np.pi
        target_values = np.array([np.pi/2, np.pi/2, -np.pi/2])
        x = target_values - estimates
        return {
            'IZ' : -x[0],
            'ZI' : -x[1],
            'ZZ' : -x[2]
        }


    @property
    def best_raw(self):
        germ = self.edesign.germs[0]    
        raws = {
            '0+' : self.raw_estimates[germ]['0+'][self.last_good_idxs[germ]['0+']],
            '1+' : self.raw_estimates[germ]['1+'][self.last_good_idxs[germ]['1+']],
            '+1' : self.raw_estimates[germ]['+1'][self.last_good_idxs[germ]['+1']],
        }
        return {key : (raws[key] + np.pi) % (2*np.pi) - np.pi  for key in raws.keys()}

