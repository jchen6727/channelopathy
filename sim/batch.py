"""
batch.py 

Batch simulation for M1 model using NetPyNE

Contributors: salvadordura@gmail.com
"""
from netpyne.batch import Batch
from netpyne import specs
import numpy as np
import os

# ----------------------------------------------------------------------------------------------
# Custom
# ----------------------------------------------------------------------------------------------
def custom(UCDAVIS=False):
    params = specs.ODict()
    # # long-range inputs
    #params[('weightLong', 'TPO')] = [0.5] #[0.25, 0.5, 0.75] 
    #params[('weightLong', 'TVL')] = [0.5] #[0.25, 0.5, 0.75] 
    #params[('weightLong', 'S1')] = [0.5] #[0.25, 0.5, 0.75] 
    #params[('weightLong', 'S2')] = [0.5] #[0.25, 0.5, 0.75] 
    #params[('weightLong', 'cM1')] = [0.5] #[0.25, 0.5, 0.75] 
    #params[('weightLong', 'M2')] = [0.5]  #[0.25, 0.5, 0.75] 
    #params[('weightLong', 'OC')] = [0.5]  #[0.25, 0.5, 0.75]	

    # EEgain
    #params['EEGain'] = [1.0] 
    params['IPTGain'] = [0.8, 0.9, 1, 1.1, 1.2]

    params['PTNaFactor'] = [0.5, 1.]


    # IEgain
    ## L2/3+4
    #params[('IEweights',0)] =  [1.0]
    ## L5
    #params[('IEweights',1)] = [1.0]   
    ## L6
    #params[('IEweights',2)] =  [1.0]  

    # IIGain
    #params['IIGain'] = [0.5, 1.0]


    #groupedParams = [('weightLong', 'TPO'), 
    #                ('weightLong', 'TVL'), 
    #                ('weightLong', 'S1'), 
    #                ('weightLong', 'S2'), 
    #                ('weightLong', 'cM1'), 
    #                ('weightLong', 'M2'), 
    #                ('weightLong', 'OC')] 
    groupedParams = []
    # --------------------------------------------------------
    # initial config
    initCfg = {}
    initCfg['duration'] = 1500
    initCfg['printPopAvgRates'] = [0, 1500] 
    initCfg['dt'] = 0.025
    initCfg['UCDAVIS'] = UCDAVIS
    if UCDAVIS: initCfg['addSubConn']=True #False
    
    #initCfg['scaleDensity'] = 1.0

    # cell params
    #initCfg['ihGbar'] = 0.75  # ih (for quiet/sponti condition)
    #initCfg['ihModel'] = 'migliore'  # ih model
    #initCfg['ihGbarBasal'] = 1.0 # multiplicative factor for ih gbar in PT cells
    #initCfg['ihlkc'] = 0.2 # ih leak param (used in Migliore)
    #initCfg['ihLkcBasal'] = 1.0 # multiplicative factor for ih lk in PT cells
    #initCfg['ihLkcBelowSoma'] = 0.01 # multiplicative factor for ih lk in PT cells
    #initCfg['ihlke'] = -86  # ih leak param (used in Migliore)
    #initCfg['ihSlope'] = 28  # ih leak param (used in Migliore)

    #initCfg['somaNa'] = 5.0  # somatic Na conduct
    #initCfg['dendNa'] = 0.3  # dendritic Na conduct (reduced to avoid dend spikes) 
    #initCfg['axonNa'] = 7   # axon Na conduct (increased to compensate) 
    #initCfg['axonRa'] = 0.005
    #initCfg['gpas'] = 0.5
    #initCfg['epas'] = 0.9
    
    # long-range input params
    #initCfg['numCellsLong'] = 1000
    #initCfg[('pulse', 'pop')] = 'None'
    #initCfg[('pulse', 'start')] = 1000.0
    #initCfg[('pulse', 'end')] = 1100.0
    #initCfg[('pulse', 'noise')] = 0.8

    # conn params
    #initCfg['IEdisynapticBias'] = None

    #initCfg['weightNormThreshold'] = 4.0
    #initCfg['IEGain'] = 1.0
    #initCfg['IPTGain'] = 1.0
    #initCfg['IIweights'] = [1.0, 1.0, 1.0]

    # plotting and saving params
    initCfg[('analysis','plotRaster','timeRange')] = initCfg['printPopAvgRates']
    initCfg[('analysis', 'plotTraces', 'timeRange')] = initCfg['printPopAvgRates']
    
    initCfg[('analysis', 'plotTraces', 'oneFigPer')] = 'trace'

    initCfg['saveCellSecs'] = False
    initCfg['saveCellConns'] = False
    
    b = Batch(params=params, netParamsFile='netParams.py', cfgFile='cfg.py', initCfg=initCfg, groupedParams=groupedParams)
    b.method = 'grid'  

    return b

# ----------------------------------------------------------------------------------------------
# Weight Normalization Exc
# ----------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------
# Evol
# ----------------------------------------------------------------------------------------------
def evolRates():
    # --------------------------------------------------------
    # parameters
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
    # --------------------------------------------------------
    # initial config


    #initCfg['scaleDensity'] = 1.0 # NOT A THING!

    # cell params


    # --------------------------------------------------------
    # fitness function
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    Epops = ['IT2', 'IT4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']

    Etune = {'target': 5, 'width': 5, 'min': 0.5}
    for pop in Epops:
        pops[pop] = Etune
    
    ## Inh pops 
    Ipops = ['PV2', 'SOM2',
            'PV5A', 'SOM5A',
            'PV5B', 'SOM5B',
            'PV6', 'SOM6']

    Itune = {'target': 10, 'width': 15, 'min': 0.25}
    for pop in Ipops:
        pops[pop] = Itune
    
    fitnessFuncArgs['pops'] = pops
    fitnessFuncArgs['maxFitness'] = 1000


    def fitnessFunc(simData, **kwargs):
        import numpy as np
        pops = kwargs['pops']
        maxFitness = kwargs['maxFitness']
        popFitness = [min(np.exp(abs(v['target'] - simData['popRates'][k])/v['width']), maxFitness) 
                if simData['popRates'][k] > v['min'] else maxFitness for k,v in pops.items()]
        fitness = np.mean(popFitness)

        popInfo = '; '.join(['%s rate=%.1f fit=%1.f'%(p, simData['popRates'][p], popFitness[i]) for i,p in enumerate(pops)])
        print('  '+popInfo)
        return fitness
    
    #from IPython import embed; embed()

    b = Batch(params=params, groupedParams=groupedParams, initCfg=initCfg)
    b.method = 'evol' 

    # Set evol alg configuration
    b.evolCfg = {
        'evolAlgorithm': 'custom',
        'fitnessFunc': fitnessFunc, # fitness expression (should read simData)
        'fitnessFuncArgs': fitnessFuncArgs,
        'pop_size': 10,
        'num_elites': 2,
        'mutation_rate': 0.5,
        'crossover': 0.5,
        'maximize': False, # maximize fitness function?
        'max_generations': 40,
        'time_sleep': 5*300, # 5min wait this time before checking again if sim is completed (for each generation)
        'maxiter_wait': 10*64, # (5h20) max number of times to check if sim is completed (for each generation)
        'defaultFitness': 1000, # set fitness value in case simulation time is over
        'scancelUser': 'ext_romanbaravalle_gmail_com'
    }


    return b


# ----------------------------------------------------------------------------------------------
# Run configurations
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# Main code
# ----------------------------------------------------------------------------------------------

if __name__ == '__main__': 

    # Figure 2 (Control Quiet) 
    # b = v56_batch7()
    
    # Figures 3, 4 (Control Quiet+Move)
    #b = v56_batch19()

    # Figure 5 (MTh Inact Quiet+Move)
    # b = v56_batch20()

    # Figure 5 (NA-R block Quiet+Move)
    # b = v56_batch22()

    # Figure 6 (VL vs Ih Quiet+Move)
    # b = v56_batch5b()

    UCDAVIS = True
    b = custom(UCDAVIS)
    if UCDAVIS:
        b.batchLabel = 'grid_M1_UCDavis_SubCellDist'
    else:
        b.batchLabel = 'grid_M1_Orig'
    b.saveFolder = '../batchSims/'+b.batchLabel
    setRunCfg(b, 'hpc_sge_evol')
    b.run() # run batch

    #b = weightNormE(pops = ['PT5B'], locs = None,
    #    allSegs = True, rule = 'PT5B_full', weights = list(np.arange(0.01, 0.2, 0.01)/100.0))

    #b.batchLabel = 'wscaleUCDavis'
    #b.saveFolder = '../batchSims/'+b.batchLabel
    #b.method = 'grid'  # evol
    #setRunCfg(b, 'hpc_sge_wscale')
    #setRunCfg(b, 'mpi_bulletin')
    #b.run() # run batch

    #b = evolRates()
    #b.batchLabel = 'evol'
    #b.saveFolder = '../batchSims/'+b.batchLabel
    #setRunCfg(b, 'hpc_sge_evol')
    #b.run() # run batch

