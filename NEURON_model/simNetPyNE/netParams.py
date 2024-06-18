from netpyne import specs, sim

import json

try:
    from __main__ import cfg
except:
    from cfg import cfg

"""
netParams
"""

netParams = specs.NetParams()


fptr = open("Na1216TF.json", "r")
hhcell = json.load(fptr)
fptr.close()

hhcell['conds'] = {'cellType': 'hh'}

netParams.cellParams['hh'] = hhcell


"""
stimulation parameters
"""

# add stim source
netParams.stimSourceParams['ic'] = {'type': 'IClamp', 'delay': 300, 'dur': 1000, 'amp': cfg.amp}
# connect stim source to target
netParams.stimTargetParams['ic->hh'] = {'source': 'ic', 'conds': {'pop': 'hh'}, 'sec': 'soma_0', 'loc': 0.5}
netParams.popParams['hh'] = {'cellType': 'hh', 'numCells': 1}
