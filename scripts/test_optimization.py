import yaml
# import numpy as np

# from CostFunctions.main import ExtractCost
# from Optimization.main import BayesianOptimization

from HIL.optimization.HIL import HIL






def run():

    args = yaml.safe_load(open('configs/test_function.yml','r'))
    hil = HIL(args)
    hil.start()


run()