from main import *
from bayes_opt import  BayesianOptimization
import matplotlib.pyplot as plt
from matplotlib import gridspec
import subprocess

if __name__  == '__main__':
    bo = BayesianOptimization(train, { '_lr': (1e-8, 0.04)}, verbose=1)
    bo.maximize(init_points=6, n_iter=10,acq= 'ei')
    f = open("res_max_val.txt","w")
    f.write(str(bo.res['max']['max_val']))
    f.close()
    f = open("res_max.txt", "w")
    f.write(str(bo.res['max']))
    f.close()
    f = open("res_all.txt", "w")
    f.write(str(bo.res['all']))
    f.close()
