from netpyne.batchtools.search import search

# run search on UCDAVIS cell.
params = {
    'IPTGain': [0.8, 0.9, 1, 1.1, 1.2],
    'PTNaFactor': [0.5, 1.], 
}

Contributors: salvadordura@gmail.com
"""
from netpyne.batchtools.search import search

weightLong = { # long-range inputs
    'weightLong.TPO': [0.25, 0.75],
    'weightLong.TVL': [0.25, 0.75],
    'weightLong.S1': [0.25, 0.75],
    'weightLong.S2': [0.25, 0.75],
    'weightLong.cM1': [0.25, 0.75],
    'weightLong.M2': [0.25, 0.75],
    'weightLong.OC': [0.25, 0.75],
}

gain = {  # EEgain
    'EEGain': [0.5, 1.5],
}

weights = {
    'IEweights.0': [0.5, 1.5],  # L2/3+4
    'IEweights.1': [0.5, 1.5],  # L5
    'IEweights.2': [0.5, 1.5],  # L6
    'IIweights.0': [0.5, 1.5],  # L2/3+4
    'IIweights.1': [0.5, 1.5],  # L5
    'IIweights.2': [0.5, 1.5],  # L6
}

params = {**weightLong, **gain, **weights}

sge_config = {
    'queue': 'cpu.q',
    'cores': 19,
    'vmem': '90G', #90G
    'realtime': '15:00:00',
    'command': 'mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi init.py'}

run_config = sge_config

search(job_type = 'sge', # or 'sh'
       comm_type = 'socket',
       label = 'optuna',
       params = params,
       output_path = '../optuna_batch',
       checkpoint_path = '../ray',
       run_config = run_config,
       num_samples = 27,
       metric = 'fitness',
       mode = 'min',
       algorithm = "optuna",
       max_concurrent = 3)



