from netpyne.batchtools.search import search

# run search on UCDAVIS cell.
params = {
    'IPTGain': [0.8, 0.9, 1, 1.1, 1.2],
    'PTNaFactor': [0.5, 1.], 
}

sge_config = {
    'queue': 'cpu.q',
    'cores': 19,
    'vmem': '90G', #90G
    'realtime': '15:00:00',
    'command': 'mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi init.py'}

run_config = sge_config

search(job_type = 'sge', # or 'sh'
       comm_type = 'socket',
       label = 'grid',

       params = params,
       output_path = '../grid_batch',
       checkpoint_path = '../ray',
       run_config = run_config,
       num_samples = 1,
       metric = 'loss',
       mode = 'min',
       algorithm = "variant_generator",
       max_concurrent = 4)
