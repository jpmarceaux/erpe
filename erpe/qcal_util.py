import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy
import yaml
import pygsti

import qcal as qc

from qcal.units import *
from qcal.utils import load_from_pickle
from qcal.backend.qubic.qpu import QubicQPU
from qcal.backend.qubic.utils import qubic_sequence

from qcal.benchmarking.readout import ReadoutFidelity
from qcal.calibration.readout import ReadoutCalibration

from qcal.interface.pygsti.circuits import load_circuits
from qcal.interface.pygsti.transpiler import Transpiler
from qcal.interface.pygsti.datasets import generate_pygsti_dataset


def make_dataset(cfg, edesign, n_shots, classifier):
    circuits = edesign.circuit_list

    transpiler = Transpiler()
    tcircuits = transpiler.transpile(circuits)

    cfg.load()

    cfg['readout/esp/enable'] = False
    qpu = QubicQPU(
        cfg,
        classifier=classifier,
        n_circs_per_seq=n_shots,
        reload_pulse=False,
        zero_between_reload=False
    )
    qpu.run(tcircuits)
    generate_pygsti_dataset(tcircuits, qpu.data_manager.save_path + 'RPE_')
    ds_location = qpu.data_manager.save_path + 'RPE_dataset.txt'
    ds = pygsti.io.load_dataset(ds_location)
    return ds

def parse_control_params_from_cfg_1qb(cfg, qid):
    with open(cfg.filename, 'r') as f:
        config_yaml = yaml.safe_load(f)
    freq = config_yaml['single_qubit'][int(qid[1])]['GE']['freq']
    phase = config_yaml['single_qubit'][int(qid[1])]['GE']['X90']['pulse'][0]['kwargs']['phase']
    phase2 = config_yaml['single_qubit'][int(qid[1])]['GE']['X90']['pulse'][2]['kwargs']['phase']
    assert phase == phase2
    amp = config_yaml['single_qubit'][int(qid[1])]['GE']['X90']['pulse'][1]['kwargs']['amp']
    return {
        'freq': freq, 
        'phase': phase, 
        'amp': amp
    }

def make_new_config_from_old(cfg, qid, config_dict, new_filename='temp.yaml'):
    """
    makes a new config file -- does not change the old
    """
    with open(cfg.filename, 'r') as f:
        config_yaml = yaml.safe_load(f)
    config_yaml['single_qubit'][int(qid[1])]['GE']['X90']['pulse'][0]['kwargs']['phase'] = float(config_dict['phase'])
    config_yaml['single_qubit'][int(qid[1])]['GE']['X90']['pulse'][2]['kwargs']['phase'] = float(config_dict['phase'])
    config_yaml['single_qubit'][int(qid[1])]['GE']['freq'] = float(config_dict['freq'])
    config_yaml['single_qubit'][int(qid[1])]['GE']['X90']['pulse'][1]['kwargs']['amp'] = float(config_dict['amp'])
    with open('configurations/'+new_filename, 'w') as f:
        yaml.dump(config_yaml, f)
    return qc.Config('configurations/'+new_filename)


def parse_cz_control_params_from_cfg(cfg, qids):
    with open(cfg.filename, 'r') as f:
        config_yaml = yaml.safe_load(f)
    qidx = [int(qid[1:]) for qid in qids]
    assert len(qids) == 2
    key_pair = f'({qidx[0]}, {qidx[1]})'
    drive_amp_q1 = config_yaml['two_qubit'][key_pair]['CZ']['pulse'][0]['kwargs']['amp']
    drive_amp_q1 = config_yaml['two_qubit'][key_pair]['CZ']['pulse'][1]['kwargs']['amp']
    assert drive_amp_q1 == drive_amp_q1
    vz_phase_q1 = config_yaml['two_qubit'][key_pair]['CZ']['pulse'][2]['kwargs']['phase']
    vz_phase_q2 = config_yaml['two_qubit'][key_pair]['CZ']['pulse'][3]['kwargs']['phase']
    return {
        'cz amp': drive_amp_q1,
        'cz vphase 1': vz_phase_q1,
        'cz vphase 2': vz_phase_q2
    }
    
def make_new_cz_config_from_old(cfg, qids, config_dict, new_filename='temp.yaml'):
    """
    makes a new config with CZ updates -- does not change the old
    """
    with open(cfg.filename, 'r') as f:
        config_yaml = yaml.safe_load(f)
    qidx = [int(qid[1:]) for qid in qids]
    key_pair = f'({qidx[0]}, {qidx[1]})'
    config_yaml['two_qubit'][key_pair]['CZ']['pulse'][0]['kwargs']['amp'] = float(config_dict['cz amp'])
    config_yaml['two_qubit'][key_pair]['CZ']['pulse'][1]['kwargs']['amp'] = float(config_dict['cz amp'])
    config_yaml['two_qubit'][key_pair]['CZ']['pulse'][2]['kwargs']['phase'] = float(config_dict['cz vphase 1'])
    config_yaml['two_qubit'][key_pair]['CZ']['pulse'][3]['kwargs']['phase'] = float(config_dict['cz vphase 2'])
    with open('configurations/'+new_filename, 'w') as f:
        yaml.dump(config_yaml, f)
    return qc.Config('configurations/'+new_filename)

def theta_cz_to_config_dict(theta_cz):
    return {'cz amp': theta_cz[0], 'cz vphase 1': theta_cz[1], 'cz vphase 2': theta_cz[2]}