from netpyne import specs

cfg = specs.SimConfig()
"""
cfg
"""
cfg.duration = 2000
cfg.dt = 0.025
cfg.verbose = False
cfg.hParams = {'celsius': 34, 'v_init': -70} #changed from -65
cfg.recordTraces = {'v': {'sec':'soma_0', 'loc':0.5, 'var':'v'}} # Dict with traces to record
cfg.recordStep = 0.1  # Step size in ms to save data
cfg.filename = 'PT_cfg'
cfg.saveJson = True
cfg.printPopAvgRates = True
cfg.analysis['plotRaster'] = {'saveFig': True}                   # Plot a raster
cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': True}  # Plot recorded traces for this list of cells

cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']


"""
batch parameter exploration
"""
cfg.amp = 0.2
