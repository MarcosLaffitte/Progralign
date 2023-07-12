################################################################################
#                                                                              #
#  README - Program: Correlation Check                                         #
#                                                                              #
################################################################################


# requires / used versions #####################################################
"""
> Lenguaje: python 3.10.9
> Anaconda: conda 23.3.1
> Packages installed with anaconda:
***** networkx 2.8.4
***** matplotlib 3.7.0
***** graphkit-learn 0.2.1
> Useful links
***** https://graphkit-learn.readthedocs.io/en/master/
***** https://github.com/jajupmochi/graphkit-learn
***** https://github.com/jajupmochi/graphkit-learn/blob/master/gklearn/utils/kernels.py
***** https://github.com/jajupmochi/graphkit-learn/blob/master/notebooks/run_structuralspkernel.ipynb
"""


# dependencies #################################################################


# installed with conda ---------------------------------------------------------
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gklearn.kernels.structuralspKernel as sspk
from matplotlib.backends.backend_pdf import PdfPages


# already in python ------------------------------------------------------------
import time
import random
from sys import argv
import pickle as pkl
from copy import deepcopy


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# variables ####################################################################


# input files ------------------------------------------------------------------
# totData = 50
theFile = None
theDict = dict()
inputFiles = argv[1:]


# data holders -----------------------------------------------------------------
pmValsK = []
pmValsS = []
mmValsK = []
mmValsS = []
timeVals = []


# output files -----------------------------------------------------------------
nameScatterPM = "scatter_PM.pdf"
nameScatterMM = "scatter_MM.pdf"
nameRunningTime = "Running_Time.pdf"


# main #########################################################################


# get data from files
for eachFile in inputFiles:
    # open each file
    theFile = open(eachFile, "rb")
    theDict = pkl.load(theFile)
    theFile.close()
    # get primal-mutant kernel distance
    pmValsK = pmValsK + theDict["KernelDistancePMVals"]
    # get primal-mutant score distance
    pmValsS = pmValsS + theDict["ScoreDistancePMVals"]
    # get mutant-mutant kernel distance
    mmValsK = mmValsK + theDict["KernelDistanceMMVals"]
    # get mutant-mutant score distance
    mmValsS = mmValsS + theDict["ScoreDistanceMMVals"]
    # get time values
    timeVals = timeVals + [theDict["Time"]]


# constructionDict["KernelDistancePMVals"] = deepcopy(kernelDistancePMVals)
# constructionDict["ScoreDistancePMVals"] = deepcopy(scoreDistancePMVals)
# constructionDict["KernelDistanceMMVals"] = deepcopy(kernelDistanceMMVals)
# constructionDict["ScoreDistanceMMVals"] = deepcopy(scoreDistanceMMVals)


# task message
print("\n")
print("*** Making scatter plot of score-distance vs kernel-distance between primal and mutants ...")


plt.scatter(pmValsK, pmValsS)
plt.title("Distance between Primal and its Mutants", fontsize = 10)
plt.xlabel("Kernel-Distance", fontsize = 9)
plt.ylabel("Score-Distance", fontsize = 9)
plt.tight_layout()
plt.savefig(nameScatterPM)
plt.close()


# task message
print("\n")
print("*** Making scatter plot of score-distance vs kernel-distance between pairs of mutants ...")


plt.scatter(mmValsK, mmValsS)
plt.title("Distance between Pairs of Mutants", fontsize = 10)
plt.xlabel("Kernel-Distance", fontsize = 9)
plt.ylabel("Score-Distance", fontsize = 9)
plt.tight_layout()
plt.savefig(nameScatterMM)
plt.close()


# task message
print("\n")
print("*** Make time histogram ...")


# plt.hist(timeVals)
plt.hist(timeVals, bins =  10)
plt.title("Running Time", fontsize = 10)
plt.xlabel("Time [s]", fontsize = 9)
plt.ylabel("Number of scenarios", fontsize = 9)
plt.tight_layout()
plt.savefig(nameRunningTime)
plt.close()



# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
