################################################################################
#                                                                              #
#  README - Program: Progralign_Pairwise_Plots.py                              #
#                                                                              #
#  --------------------------------------------------------------------------  #
#                                                                              #
# - LICENSE:                                                                   #
#                                                                              #
#   This file is part of the work published in                                 #
#            [article url]                                                     #
#   and it is released under                                                   #
#            MIT License Copyright (c) 2023 Marcos E. González Laffitte        #
#   See LICENSE file in                                                        #
#            https://github.com/MarcosLaffitte/Progralign                      #
#   for full license details.                                                  #
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


# already in python ------------------------------------------------------------
import time
import random
from sys import argv
import pickle as pkl
from copy import deepcopy
from statistics import mean
from math import modf, sqrt
from operator import eq, itemgetter
from itertools import product, combinations


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# input and variables ##########################################################


# input ------------------------------------------------------------------------
inputFileName = ""
inputFileName = argv[1]
inputList = []
inputFile = None
drawingMode = False


# data holders -----------------------------------------------------------------
# expand
average_time_expand = 0
resultsPerCell_expand = []
allValues_time_expand = []
tempArray_time_expand = []
runningTime_expand = dict()
average_mcs_expand = 0
allValues_mcs_expand = []
tempArray_mcs_expand = []
orderMCS_expand = dict()
valuesByOrder_expand = []
valuesByDensity_expand = []
timeData_expand = []
# trimm
average_time_trimm = 0
resultsPerCell_trimm = []
allValues_time_trimm = []
tempArray_time_trimm = []
runningTime_trimm = dict()
average_mcs_trimm = 0
allValues_mcs_trimm = []
tempArray_mcs_trimm = []
orderMCS_trimm = dict()
valuesByOrder_trimm = []
valuesByDensity_trimm = []
timeData_trimm = []
# order MCS
valuesByOrder_MCS = []
valuesByDensity_MCS = []


# parameters -------------------------------------------------------------------
orderValues = [8, 9, 10, 11, 12, 13, 14, 15, 16]
densityPairs = [(25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70)]
densityTicks = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
densityPairsStr = ["(25, 30)", "(30, 35)", "(35, 40)", "(40, 45)", "(45, 50)", "(50, 55)", "(55, 60)", "(60, 65)", "(65, 70)"]


# main #########################################################################


# initial message
print("\n")
print(">>> Progralign Pairwise Plots - Progralign Github Repository")


# retrieve input file
inputFile = open(inputFileName, "rb")
inputTuple = pkl.load(inputFile)
inputFile.close()


# unpack data
pairwiseAlignment_expand = deepcopy(inputTuple[0])
pairwiseAlignment_trimm = deepcopy(inputTuple[1])
allIndices = deepcopy(inputTuple[2])
allGraphs = deepcopy(inputTuple[3])
existence = deepcopy(inputTuple[4])
graphsByCells = deepcopy(inputTuple[5])


# build data frames with results
allValues_time_expand = []
allValues_time_trimm = []
allValues_mcs_expand = []
allValues_mcs_trimm = []
allValues_mean_mcs_expand = []
allValues_mean_mcs_trimm = []
for eachOrder in orderValues:
    # reinitialize array
    tempArray_time_expand = []
    tempArray_time_trimm = []
    tempArray_mcs_expand = []
    tempArray_mcs_trimm = []
    # get average runnig time for each density interval
    for eachDensityPair in densityPairs:
        # data of expand
        resultsPerCell_expand = deepcopy(pairwiseAlignment_expand[(eachOrder, eachDensityPair)])
        average_time_expand = mean([eachRT for (i1, i2, orderMCS, eachRT) in resultsPerCell_expand])
        tempArray_time_expand.append(average_time_expand)
        average_mcs_expand = mean([orderMCS for (i1, i2, orderMCS, eachRT) in resultsPerCell_expand])
        tempArray_mcs_expand.append(average_mcs_expand)
        allValues_mcs_expand = allValues_mcs_expand + [orderMCS for (i1, i2, orderMCS, eachRT) in resultsPerCell_expand]
        # data of trimm
        resultsPerCell_trimm = deepcopy(pairwiseAlignment_trimm[(eachOrder, eachDensityPair)])
        average_time_trimm = mean([eachRT for (i1, i2, orderMCS, eachRT) in resultsPerCell_trimm])
        tempArray_time_trimm.append(average_time_trimm)
        allValues_time_trimm = allValues_time_trimm + tempArray_time_trimm
        average_mcs_trimm = mean([orderMCS for (i1, i2, orderMCS, eachRT) in resultsPerCell_trimm])
        tempArray_mcs_trimm.append(average_mcs_trimm)
        allValues_mcs_trimm = allValues_mcs_trimm + [orderMCS for (i1, i2, orderMCS, eachRT) in resultsPerCell_trimm]
    # save array of averages
    runningTime_expand[eachOrder] = deepcopy(tempArray_time_expand)
    runningTime_trimm[eachOrder] = deepcopy(tempArray_time_trimm)
    orderMCS_expand[eachOrder] = deepcopy(tempArray_mcs_expand)
    orderMCS_trimm[eachOrder] = deepcopy(tempArray_mcs_expand)
    allValues_mean_mcs_expand = allValues_mean_mcs_expand + tempArray_mcs_expand
    allValues_mean_mcs_trimm = allValues_mean_mcs_trimm + tempArray_mcs_trimm


# consistency of values
for i in range(len(allValues_mcs_expand)):
    if(not allValues_mcs_expand[i] == allValues_mcs_trimm[i]):
        print("NOOOOOO", allValues_mcs_expand[i], allValues_mcs_trimm[i])


# draw heatmap of running time with expand
myCmap = plt.cm.Blues
# create color normalization
minValue = 0
maxValue = 0.7
myNorm = plt.Normalize(vmin = minValue, vmax = maxValue)
# create data frame
finalArray = [runningTime_expand[eachOrder] for eachOrder in orderValues]
maskedArray = np.ma.array(finalArray)
maskedArray = np.transpose(maskedArray)
im = plt.imshow(maskedArray, cmap = myCmap, norm = myNorm, aspect = "equal")
# set figure attributes
ax = plt.gca()
# set ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticks(list(range(len(orderValues))), labels = orderValues, fontsize = 9)
ax.set_xticks([val-0.5 for val in range(0, len(orderValues)+1)], fontsize = 9, minor = True)
ax.set_yticks([val-0.5 for val in range(0, 10)], labels = densityTicks, fontsize = 9, minor = True)
# set grid
ax.grid(which = "minor", color = "w", linestyle = "-", linewidth = 3)
# remove minor ticks from x axis
ax.tick_params(which = "minor", bottom = False)
# move major ticks and labels of x axis to top
ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
# finish image
plt.xlabel("VF2_step", fontsize = 12)#, weight = "light")
plt.title("Order\n", fontsize = 12)#, weight = "light")
plt.ylabel("Intervals of proportion of edges\n", fontsize = 12)#, weight = "light")
# color bar
cbar = ax.figure.colorbar(im, ax = ax)
cbar.set_ticks([])
cbar.set_ticks([val/10 for val in range(0, 8)])
cbar.set_ticklabels([val/10 for val in range(0, 8)])
cbar.ax.tick_params(labelsize = 8)
cbar.ax.set_ylabel("Average running time [s]", rotation = -90, fontsize = 12, va = "bottom")#, weight = "light")
# save figure
plt.tight_layout()
plt.savefig("RunningTime_Expand.pdf")
plt.close()


# draw heatmap of running time with trimm
myCmap = plt.cm.Greens
# create color normalization
minValue = min(allValues_time_trimm)
maxValue = max(allValues_time_trimm)
myNorm = plt.Normalize(vmin = minValue, vmax = maxValue)
# create data frame
finalArray = [runningTime_trimm[eachOrder] for eachOrder in orderValues]
maskedArray = np.ma.array(finalArray)
maskedArray = np.transpose(maskedArray)
im = plt.imshow(maskedArray, cmap = myCmap, norm = myNorm, aspect = "equal")
# set figure attributes
ax = plt.gca()
# set ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticks(list(range(len(orderValues))), labels = orderValues, fontsize = 9)
ax.set_xticks([val-0.5 for val in range(0, len(orderValues)+1)], fontsize = 9, minor = True)
ax.set_yticks([val-0.5 for val in range(0, 10)], labels = densityTicks, fontsize =  9, minor = True)
# set grid
ax.grid(which = "minor", color = "w", linestyle = "-", linewidth = 3)
# remove minor ticks from x axis
ax.tick_params(which = "minor", bottom = False)
# move major ticks and labels of x axis to top
ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
# finish image
plt.xlabel("Iterative_Trimming", fontsize = 12)#, weight = "light")
plt.title("Order\n", fontsize = 12)#, weight = "light")
plt.ylabel("Intervals of proportion of edges\n", fontsize = 12)#, weight = "light")
# color bar
cbar = ax.figure.colorbar(im, ax = ax)
cbar.set_ticks([])
cbar.set_ticks([val*5 for val in range(0, 8)])
cbar.set_ticklabels([val*5 for val in range(0, 8)])
cbar.ax.tick_params(labelsize = 8)
cbar.ax.set_ylabel("Average running time [s]", rotation = -90, fontsize = 12, va = "bottom")#, weight = "light")
# save figure
plt.tight_layout()
plt.savefig("RunningTime_Trimm.pdf")
plt.close()


# draw heatmap of order of MCS
myCmap = plt.cm.Reds
# create color normalization
minValue = min(allValues_mean_mcs_expand)-0.2
maxValue = max(allValues_mean_mcs_expand)+0.2
myNorm = plt.Normalize(vmin = minValue, vmax = maxValue)
# create data frame
finalArray = [orderMCS_expand[eachOrder] for eachOrder in orderValues]
maskedArray = np.ma.array(finalArray)
maskedArray = np.transpose(maskedArray)
im = plt.imshow(maskedArray, cmap = myCmap, norm = myNorm)
# set figure attributes
ax = plt.gca()
# set ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticks(list(range(len(orderValues))), labels = orderValues, fontsize = 9)
ax.set_xticks([val-0.5 for val in range(0, len(orderValues)+1)], fontsize = 9, minor = True)
ax.set_yticks([val-0.5 for val in range(0, 10)], labels = densityTicks, fontsize = 9, minor = True)
# set grid
ax.grid(which = "minor", color = "w", linestyle = "-", linewidth = 3)
# remove minor ticks from x axis
ax.tick_params(which = "minor", bottom = False)
# move major ticks and labels of x axis to top
ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
# finish image
plt.xlabel("Order of MCIS between subsets of random graphs", fontsize = 12)#, weight = "light")
plt.title("Order\n", fontsize = 12)#, weight = "light")
plt.ylabel("Intervals of proportion of edges\n", fontsize = 12)#, weight = "light")
# color bar
cbar = ax.figure.colorbar(im, ax = ax)
cbar.ax.tick_params(labelsize = 8)
cbar.ax.set_ylabel("Average order of MCIS", rotation = -90, fontsize = 12, va = "bottom")#, weight = "light")
# save figure
plt.tight_layout()
plt.savefig("MCIS_Order.pdf")
plt.close()


# build data frames with results for boxplot
valuesByOrder_trimm = []
valuesByOrder_expand = []
valuesByOrder_MCS = []
valuesByDensity_trimm = []
valuesByDensity_expand = []
valuesByDensity_MCS = []
for eachOrder in orderValues:
    # reinitialize temp array
    tempArray_expand = []
    tempArray_trimm = []
    tempArray_MCS = []
    # get average runnig time for each density interval
    for eachDensityPair in densityPairs:
        # data of expand
        resultsPerCell_expand = deepcopy(pairwiseAlignment_expand[(eachOrder, eachDensityPair)])
        timeData_expand = [eachRT for (i1, i2, orderMCS, eachRT) in resultsPerCell_expand]
        tempArray_expand = tempArray_expand + timeData_expand
        # data of trimm
        resultsPerCell_trimm = deepcopy(pairwiseAlignment_trimm[(eachOrder, eachDensityPair)])
        timeData_trimm = [eachRT for (i1, i2, orderMCS, eachRT) in resultsPerCell_trimm]
        tempArray_trimm = tempArray_trimm + timeData_trimm
        # data of MCS
        orderData = [orderMCS for (i1, i2, orderMCS, eachRT) in resultsPerCell_expand]
        tempArray_MCS = tempArray_MCS + orderData
    # save data for boxplot
    valuesByOrder_expand.append(tempArray_expand)
    valuesByOrder_trimm.append(tempArray_trimm)
    valuesByOrder_MCS.append(tempArray_MCS)


for eachDensityPair in densityPairs:
    # reinitialize temp array
    tempArray_expand = []
    tempArray_trimm = []
    tempArray_MCS = []
    # get average runnig time for each density interval
    for eachOrder in orderValues:
        # data of expand
        resultsPerCell_expand = deepcopy(pairwiseAlignment_expand[(eachOrder, eachDensityPair)])
        timeData_expand = [eachRT for (i1, i2, orderMCS, eachRT) in resultsPerCell_expand]
        tempArray_expand = tempArray_expand + timeData_expand
        # data of trimm
        resultsPerCell_trimm = deepcopy(pairwiseAlignment_trimm[(eachOrder, eachDensityPair)])
        timeData_trimm = [eachRT for (i1, i2, orderMCS, eachRT) in resultsPerCell_trimm]
        tempArray_trimm = tempArray_trimm + timeData_trimm
        # data of MCS
        orderData = [orderMCS for (i1, i2, orderMCS, eachRT) in resultsPerCell_expand]
        tempArray_MCS = tempArray_MCS + orderData
    # save data for boxplot
    valuesByDensity_expand.append(tempArray_expand)
    valuesByDensity_trimm.append(tempArray_trimm)
    valuesByDensity_MCS.append(tempArray_MCS)


# boxplots by EXPAND
fig, (ax1, ax2) = plt.subplots(2, 1)
violin_parts = ax1.violinplot(valuesByOrder_expand, showmeans = True)
for pc in violin_parts["bodies"]:
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = violin_parts[partname]
        vp.set_linewidth(1)
ax1.set_xticks([])
ax1.set_xticks(list(range(1, len(orderValues)+1)), labels = orderValues, fontsize = 9)
ax1.grid(linestyle = "--", linewidth = 0.6, b = True)
ax1.set_xlabel("Order", fontsize = 12, labelpad = 10)#, weight = "light")
ax1.set_ylabel("Running time [s]", fontsize = 12, labelpad = 10)#, weight = "light")
axp = ax1.twinx()
axp.set_yticks([])
violin_parts = ax2.violinplot(valuesByDensity_expand, showmeans = True)
for pc in violin_parts["bodies"]:
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = violin_parts[partname]
        vp.set_linewidth(1)
ax2.set_xticks([])
ax2.set_xticks(list(range(1, len(densityPairsStr)+1)), labels = densityPairsStr, fontsize = 8)
ax2.grid(linestyle = "--", linewidth = 0.6, b = True)
ax2.set_xlabel("Intervals of proportion of edges", fontsize = 12, labelpad = 10)#, weight = "light")
ax2.set_ylabel("Running time [s]", fontsize = 12, labelpad = 10)#, weight = "light")
axp = ax2.twinx()
axp.set_yticks([])
# save figure
fig.suptitle("Running Time over Random Graphs - VF2_step", fontsize = 12)#, weight = "light")
plt.tight_layout()
plt.savefig("RunningTime_Collapsed_Expand.pdf")
plt.close()


# boxplots by TRIMM
fig, (ax1, ax2) = plt.subplots(2, 1)
violin_parts = ax1.violinplot(valuesByOrder_trimm, showmeans = True)
for pc in violin_parts["bodies"]:
    pc.set_color("tab:green")
    pc.set_facecolor("tab:green")
    pc.set_edgecolor("tab:green")
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = violin_parts[partname]
        vp.set_edgecolor("tab:green")
        vp.set_linewidth(1)
ax1.set_xticks([])
ax1.set_xticks(list(range(1, len(orderValues)+1)), labels = orderValues, fontsize = 9)
ax1.grid(linestyle = "--", linewidth = 0.6, b = True)
ax1.set_xlabel("Order", fontsize = 12, labelpad = 10)#, weight = "light")
ax1.set_ylabel("Running time [s]", fontsize = 12, labelpad = 10)#, weight = "light")
axp = ax1.twinx()
axp.set_yticks([])
violin_parts = ax2.violinplot(valuesByDensity_trimm, showmeans = True)
for pc in violin_parts["bodies"]:
    pc.set_color("tab:green")
    pc.set_facecolor("tab:green")
    pc.set_edgecolor("tab:green")
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = violin_parts[partname]
        vp.set_edgecolor("tab:green")
        vp.set_linewidth(1)
ax2.set_xticks([])
ax2.set_xticks(list(range(1, len(densityPairsStr)+1)), labels = densityPairsStr, fontsize = 8)
ax2.grid(linestyle = "--", linewidth = 0.6, b = True)
ax2.set_xlabel("Intervals of proportion of edges", fontsize = 12, labelpad = 10)#, weight = "light")
ax2.set_ylabel("Running time [s]", fontsize = 12, labelpad = 10)#, weight = "light")
axp = ax2.twinx()
axp.set_yticks([])
# save figure
fig.suptitle("Running Time over Random Graphs - Iterative_Trimming", fontsize = 12)#, weight = "light")
plt.tight_layout()
plt.savefig("RunningTime_Collapsed_Trimm.pdf")
plt.close()


# boxplots order MCS
fig, (ax1, ax2) = plt.subplots(2, 1)
violin_parts = ax1.violinplot(valuesByOrder_MCS, showmeans = True)
for pc in violin_parts["bodies"]:
    pc.set_color("tab:red")
    pc.set_facecolor("tab:red")
    pc.set_edgecolor("tab:red")
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = violin_parts[partname]
        vp.set_edgecolor("tab:red")
        vp.set_linewidth(1)
ax1.set_xticks([])
ax1.set_xticks(list(range(1, len(orderValues)+1)), labels = orderValues, fontsize = 9)
ax1.grid(linestyle = "--", linewidth = 0.6, b = True)
ax1.set_xlabel("Order", fontsize = 12, labelpad = 10)#, weight = "light")
ax1.set_ylabel("Order of MCIS", fontsize = 12, labelpad = 10)#, weight = "light")
axp = ax1.twinx()
axp.set_yticks([])
violin_parts = ax2.violinplot(valuesByDensity_MCS, showmeans = True)
for pc in violin_parts["bodies"]:
    pc.set_color("tab:red")
    pc.set_facecolor("tab:red")
    pc.set_edgecolor("tab:red")
    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = violin_parts[partname]
        vp.set_edgecolor("tab:red")
        vp.set_linewidth(1)
ax2.set_xticks([])
ax2.set_xticks(list(range(1, len(densityPairsStr)+1)), labels = densityPairsStr, fontsize = 8)
ax2.grid(linestyle = "--", linewidth = 0.6, b = True)
ax2.set_xlabel("Intervals of proportion of edges", fontsize = 12, labelpad = 10)#, weight = "light")
ax2.set_ylabel("Order of MCIS", fontsize = 12, labelpad = 10)#, weight = "light")
axp = ax2.twinx()
axp.set_yticks([])
# save figure
fig.suptitle("Order of MCIS between subsets of random graphs", fontsize = 12)#, weight = "light")
plt.tight_layout()
plt.savefig("OrderMCIS_Collapsed.pdf")
plt.close()


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
