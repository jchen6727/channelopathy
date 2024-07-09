"""
init.py

Starting script to run NetPyNE-based M1 model.

Usage:
    python init.py # Run simulation, optionally plot a raster

MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py

Contributors: salvadordura@gmail.com
"""
from netpyne.batchtools import specs
from netpyne.batchtools import comm
from netpyne import sim
from netParams import netParams, cfg
import json
import numpy

def calculate_fitness(freq, target_freq, width, min_freq, max_freq, max_fitness):
    if freq < min_freq or freq > max_freq:
        return max_fitness
    else:
        return min(numpy.exp(abs(freq - target_freq)/width), max_fitness)


# -----------------------------------------------------------
# Main code
sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  # create network object and set cfg and net params

sim.pc.timeout(300)                          # set nrn_timeout threshold to X sec (max time allowed without increasing simulation time, t; 0 = turn off)
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations
sim.net.connectCells()            			# create connections between cells based on params
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)

# Simulation option 1: standard
sim.runSim()                              # run parallel Neuron simulation (calling func to modify mechs)


# Gather/save data option 1: standard
sim.gatherData() # should have data in sim.allSimData()

# Gather/save data option 2: distributed saving across nodes 
#sim.saveDataInNodes()
#sim.gatherDataFromFiles()

sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#

if comm.is_host():
    netParams.save("{}/{}_params.json".format(cfg.saveFolder, cfg.simLabel))
    print('transmitting data...')
    inputs = specs.get_mappings()
    #print(json.dumps({**inputs}))
    results = sim.analysis.popAvgRates(show=False)
    pop_fitness = []
    for epop in ['IT2', 'IT4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']:
        pop_fitness.append(calculate_fitness(results[epop], 5, 5, 0.5, 100, 1000))
    for ipop in ['PV2', 'SOM2', 'PV5A', 'SOM5A', 'PV5B', 'SOM5B', 'PV6', 'SOM6']:
        pop_fitness.append(calculate_fitness(results[ipop], 10, 15, 0.25, 100, 1000))
    fitness = numpy.mean(pop_fitness)
    out_json = json.dumps({'fitness': fitness})
    comm.send(out_json)
    comm.close()
