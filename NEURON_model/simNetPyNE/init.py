from netpyne import sim

# read cfg and netParams from command line arguments if available; otherwise use default
cfg, netParams = sim.readCmdLineArgs(simConfigDefault='cfg.py', netParamsDefault='netParams.py')

# Create network and run simulation
sim.initialize( simConfig = cfg, netParams = netParams)

# Create network and run simulation
sim.createSimulateAnalyze(netParams=netParams, simConfig=cfg)
