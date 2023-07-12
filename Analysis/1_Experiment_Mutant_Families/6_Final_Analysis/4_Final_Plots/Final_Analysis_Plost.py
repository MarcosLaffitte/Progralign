################################################################################
#                                                                              #
#  README - Program: Final_Analysis_Plots.pkl                                  #
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
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gklearn.kernels.structuralspKernel as sspk
from matplotlib.backends.backend_pdf import PdfPages


# already in python ------------------------------------------------------------
import time
import random
from sys import argv
import pickle as pkl
from copy import deepcopy
from math import factorial, modf, log10, sqrt
from statistics import mean


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# variables ####################################################################


# input files ------------------------------------------------------------------
totData = 50
inputFiles = ["Mutants_" + str(i) + "_Results.pkl" for i in range(1, totData+1)]
inputTuple = ()
inputDict_kGT_expand_AE = dict()
inputDict_kGT_expand_nAE = dict()
inputDict_rGT_expand_AE = dict()
inputDict_rGT_expand_nAE = dict()
inputDict_kGT_trimm_AE = dict()
inputDict_kGT_trimm_nAE = dict()
inputDict_rGT_trimm_AE = dict()
inputDict_rGT_trimm_nAE = dict()


# data holders -----------------------------------------------------------------
# boxplot of running time
timeVals_kGT_expand_AE = []
timeVals_rGT_expand_AE = []
timeVals_kGT_trimm_AE = []
timeVals_rGT_trimm_AE = []
timeVals_kGT_expand_nAE = []
timeVals_rGT_expand_nAE = []
timeVals_kGT_trimm_nAE = []
timeVals_rGT_trimm_nAE = []
# scatter plots of gaps vs ambiguous edges
gaps_inputDict_kGT_expand_AE = []
ambs_inputDict_kGT_expand_AE = []
gaps_inputDict_rGT_expand_AE = []
ambs_inputDict_rGT_expand_AE = []
gaps_inputDict_kGT_trimm_AE = []
ambs_inputDict_kGT_trimm_AE = []
gaps_inputDict_rGT_trimm_AE = []
ambs_inputDict_rGT_trimm_AE = []
gaps_inputDict_kGT_expand_nAE = []
ambs_inputDict_kGT_expand_nAE = []
gaps_inputDict_rGT_expand_nAE = []
ambs_inputDict_rGT_expand_nAE = []
gaps_inputDict_kGT_trimm_nAE = []
ambs_inputDict_kGT_trimm_nAE = []
gaps_inputDict_rGT_trimm_nAE = []
ambs_inputDict_rGT_trimm_nAE = []
GAPS = []
AMBS = []
# scatter plots of kernel-distance vs score-distance
pmValsK = []
pmValsS = []
mmValsK = []
mmValsS = []


# output files -----------------------------------------------------------------
# boxplot of running time
nameRunningTime = "1_boxplot_running_time.pdf"
# boxplot primal vs consensus
namePrimalVSConsensus = "2_lineplots_primal_vs_consensus.pdf"
namePrimalVSConsensus_Dist = "8_lineplots_primal_vs_consensus_distance.pdf"
namePrimalVSConsensus_Ord = "9_lineplots_primal_vs_consensus_order.pdf"
# scatter plots of gaps vs ambiguous edges
nameScatterGapsAmbs = "3_scatter_gaps_vs_ambiguous_edges.pdf"
nameHistGapsKernelGT = "10_histograms_gaps_kernel_GT.pdf"
nameHistGapsRandomGT = "11_histograms_gaps_random_GT.pdf"
# scatter plots of distance between primals and mutants and pairs of mutants
nameScatterPM = "4_scatter_PM.pdf"
nameScatterMM = "5_scatter_MM.pdf"
# histograms of important times
nameHistTime_Expand = "6_Histograms_Expand.pdf"
nameHistTime_Trimm = "7_Histograms_Trimm.pdf"
# histograms of order of MCS
nameHistMCS_PM = "12_Histograms_MCS_PM.pdf"
nameHistMCS_MM = "13_Histograms_MCS_MM.pdf"
# boxplot primal vs consensus with AE
namePrimalVSConsensus_AE = "14_lineplots_primal_vs_consensus_AE.pdf"
namePrimalVSConsensus_Dist_AE = "15_lineplots_primal_vs_consensus_distance_AE.pdf"
namePrimalVSConsensus_Ord_AE = "16_lineplots_primal_vs_consensus_order_AE.pdf"


# main #########################################################################


# plots to make ----------------------------------------------------------------
makePlots = dict()
makePlots["BoxplotsRTs"] = False
makePlots["LineplotsMCS"] = True
makePlots["ScatterGaps"] = True
makePlots["ScatterDist"] = True
makePlots["HistogramDist"] = False



# boxplots: running time -------------------------------------------------------
if(makePlots["BoxplotsRTs"]):
    # task message
    print("\n")
    print("* Making running time boxplots ...")
    # get data from files
    for eachFile in inputFiles:
        # open file
        inputFile = open(eachFile, "rb")
        inputTuple = pkl.load(inputFile)
        inputFile.close()
        # unpack dictionaries
        inputDict_kGT_expand_AE = inputTuple[0]
        inputDict_kGT_expand_nAE = inputTuple[1]
        inputDict_rGT_expand_AE = inputTuple[2]
        inputDict_rGT_expand_nAE = inputTuple[3]
        inputDict_kGT_trimm_AE = inputTuple[4]
        inputDict_kGT_trimm_nAE = inputTuple[5]
        inputDict_rGT_trimm_AE = inputTuple[6]
        inputDict_rGT_trimm_nAE = inputTuple[7]
        # get required data
        timeVals_kGT_expand_AE.append(inputDict_kGT_expand_AE["Time"])
        timeVals_rGT_expand_AE.append(inputDict_rGT_expand_AE["Time"])
        timeVals_kGT_trimm_AE.append(inputDict_kGT_trimm_AE["Time"])
        timeVals_rGT_trimm_AE.append(inputDict_rGT_trimm_AE["Time"])
        timeVals_kGT_expand_nAE.append(inputDict_kGT_expand_nAE["Time"])
        timeVals_rGT_expand_nAE.append(inputDict_rGT_expand_nAE["Time"])
        timeVals_kGT_trimm_nAE.append(inputDict_kGT_trimm_nAE["Time"])
        timeVals_rGT_trimm_nAE.append(inputDict_rGT_trimm_nAE["Time"])
    # obtain logarithms
    timeVals_kGT_trimm_nAE_log = [log10(t) for t in list(timeVals_kGT_trimm_nAE)]
    timeVals_rGT_trimm_nAE_log = [log10(t) for t in list(timeVals_rGT_trimm_nAE)]
    timeVals_kGT_trimm_AE_log = [log10(t) for t in list(timeVals_kGT_trimm_AE)]
    timeVals_rGT_trimm_AE_log = [log10(t) for t in list(timeVals_rGT_trimm_AE)]
    timeVals_kGT_expand_nAE_log = [log10(t) for t in list(timeVals_kGT_expand_nAE)]
    timeVals_rGT_expand_nAE_log = [log10(t) for t in list(timeVals_rGT_expand_nAE)]
    timeVals_kGT_expand_AE_log = [log10(t) for t in list(timeVals_kGT_expand_AE)]
    timeVals_rGT_expand_AE_log = [log10(t) for t in list(timeVals_rGT_expand_AE)]
    timeData = [timeVals_kGT_trimm_nAE_log,
                timeVals_kGT_trimm_AE_log,
                timeVals_rGT_trimm_nAE_log,
                timeVals_rGT_trimm_AE_log,
                timeVals_kGT_expand_nAE_log,
                timeVals_kGT_expand_AE_log,
                timeVals_rGT_expand_nAE_log,
                timeVals_rGT_expand_AE_log]
    timeLabels = [r"$Tk_{\bigcirc}$",    # kGT_Trimm_nAE"
                  r"$Tk_{\blacksquare}$",    # kGT_Trimm_AE"
                  r"$Tr_{\bigcirc}$",    # rGT_Trimm_nAE"
                  r"$Tr_{\blacksquare}$",    # rGT_Trimm_AE"
                  r"$Sk_{\bigcirc}$",    # kGT_Exp_nAE"
                  r"$Sk_{\blacksquare}$",    # kGT_Exp_AE"
                  r"$Sr_{\bigcirc}$",    # rGT_Exp_nAE"
                  r"$Sr_{\blacksquare}$"]    # rGT_Exp_AE"
    colorData = ["darkgreen",
                 "darkgreen",
                 "tab:green",
                 "tab:green",
                 "tab:blue",
                 "tab:blue",
                 "cornflowerblue",
                 "cornflowerblue"]
    whiskers = 0.5
    widthsBoxes = 0.4
    # plot boxpölots
    fig, ax = plt.subplots()
    bps = ax.boxplot(timeData, whis = whiskers, widths = widthsBoxes, labels = timeLabels, sym = ".", patch_artist = True)
    # set colors
    for i in range(len(bps["fliers"])):
        bps["fliers"][i].set(markeredgecolor = "k", markerfacecolor = colorData[i])
    for i in range(len(bps["caps"])):
        bps["caps"][i].set(color = "k")
    for i in range(len(bps["boxes"])):
        bps["boxes"][i].set(facecolor = colorData[i], edgecolor = "k")
    for i in range(len(bps["medians"])):
        bps["medians"][i].set(color = "k")
    for i in range(len(bps["whiskers"])):
        bps["whiskers"][i].set(color = "k")
    # set some attributes and save figures
    custom_lines = [Line2D([0], [0], color = "darkgreen", lw = 5),
                    Line2D([0], [0], color = "tab:green", lw = 5),
                    Line2D([0], [0], color = "tab:blue", lw = 5),
                    Line2D([0], [0], color = "cornflowerblue", lw = 5),
                    Line2D([0], [0], color = "w", lw = 0),
                    Line2D([0], [0], color = "w", lw = 0)]
    # plt.legend(custom_lines, [r"$Tk$ - Iterative_Trimming and kernel-based GT",
    #                                               r"$Tr$ - Iterative_Trimming and random GT",
    #                                               r"$Sk$ - VF2_Step and kernel-based GT",
    #                                               r"$Sr$ - VF2_Step and random GT",
    #                                               r"$\bigcirc$ - alignments without ambiguous edges",
    #                                               r"$\blacksquare$ - alignments with ambiguous edges"],
    #            fontsize = 9)
    plt.ylabel(r"Log$_{10}$" + " of running time [s]\n", fontsize = 14)#, weight = "light")
    plt.xlabel("\nExperiments", fontsize = 14)#, weight = "light")
    plt.xticks(fontsize = 12)#, weight = "light")
    plt.yticks(fontsize = 12)
    plt.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    plt.tight_layout()
    plt.savefig(nameRunningTime)
    plt.close()
    # plot histogram expand - kGT - no ambiguous edges
    fig, ax = plt.subplots()
    mean_t_expand = mean(timeVals_kGT_expand_nAE)
    plt.hist(timeVals_kGT_expand_nAE, bins = 18, rwidth = 0.98, alpha = 0.5, color = "steelblue")
    plt.vlines(mean_t_expand, color = "tab:red", ymin = 0, ymax = 12.5,
               linewidth = 2, linestyle = "--",
               label = "mean: " + str(round(mean_t_expand, 2)) + " s")
    plt.legend(fontsize = 12)
    plt.ylim(0, 12.5)
    plt.xlim(0, 150)
    plt.xlabel("Running time of VF2_step [s],\n with kernel-based GT and without ambiguous edges", fontsize = 14)#, weight = "light")
    plt.ylabel("Number of scenarios", fontsize = 14)#, weight = "light")
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    ax.set_axisbelow(True)
    plt.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    plt.tight_layout()
    plt.savefig(nameHistTime_Expand)
    plt.close()
    # plot histogram trimmm - kGT - no ambiguous edges
    fig, ax = plt.subplots()
    mean_t_trimm = mean(timeVals_kGT_trimm_nAE)
    plt.hist(timeVals_kGT_trimm_nAE, bins = 28, rwidth = 0.98, alpha = 0.5, color = "darkgreen")
    plt.vlines(mean_t_trimm, color = "tab:red", ymin = 0, ymax = 21,
               linewidth = 2, linestyle = "--",
               label = "mean: " + str(round(mean_t_trimm, 2)) + " s")
    plt.legend(fontsize = 12)
    plt.ylim(0, 21)
    plt.xlim(0, 150)
    plt.xlabel("Running time of Iterative_Trimming [s],\n with kernel-based GT and without ambiguous edges", fontsize = 14)#, weight = "light")
    plt.ylabel("Number of scenarios", fontsize = 14)#, weight = "light")
    plt.xticks(fontsize = 12)
    plt.yticks([], fontsize = 12)
    A = list(range(0, 20+1, 2))
    plt.yticks(A, A)
    ax.set_axisbelow(True)
    plt.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    plt.tight_layout()
    plt.savefig(nameHistTime_Trimm)
    plt.close()




# boxplots: order of MCS between primal and consensus graphs -------------------
# distance
distMCS_kGT_expand_AE_av = []
distMCS_rGT_expand_AE_av = []
distMCS_kGT_trimm_AE_av = []
distMCS_rGT_trimm_AE_av = []
distMCS_kGT_expand_nAE_av = []
distMCS_rGT_expand_nAE_av = []
distMCS_kGT_trimm_nAE_av = []
distMCS_rGT_trimm_nAE_av = []
# order consensus
consOrd_kGT_expand_AE_av = []
consOrd_rGT_expand_AE_av = []
consOrd_kGT_trimm_AE_av = []
consOrd_rGT_trimm_AE_av = []
consOrd_kGT_expand_nAE_av = []
consOrd_rGT_expand_nAE_av = []
consOrd_kGT_trimm_nAE_av = []
consOrd_rGT_trimm_nAE_av = []
# order MCS
orderMCS_kGT_expand_AE_av = []
orderMCS_rGT_expand_AE_av = []
orderMCS_kGT_trimm_AE_av = []
orderMCS_rGT_trimm_AE_av = []
orderMCS_kGT_expand_nAE_av = []
orderMCS_rGT_expand_nAE_av = []
orderMCS_kGT_trimm_nAE_av = []
orderMCS_rGT_trimm_nAE_av = []
# cluster distance data
distMCS_kGT_expand_AE = dict()
distMCS_rGT_expand_AE = dict()
distMCS_kGT_trimm_AE = dict()
distMCS_rGT_trimm_AE = dict()
distMCS_kGT_expand_nAE = dict()
distMCS_rGT_expand_nAE = dict()
distMCS_kGT_trimm_nAE = dict()
distMCS_rGT_trimm_nAE = dict()
# cluster consensus data
consOrd_kGT_expand_AE = dict()
consOrd_rGT_expand_AE = dict()
consOrd_kGT_trimm_AE = dict()
consOrd_rGT_trimm_AE = dict()
consOrd_kGT_expand_nAE = dict()
consOrd_rGT_expand_nAE = dict()
consOrd_kGT_trimm_nAE = dict()
consOrd_rGT_trimm_nAE = dict()
# cluster order of MCS data
orderMCS_kGT_expand_AE = dict()
orderMCS_rGT_expand_AE = dict()
orderMCS_kGT_trimm_AE = dict()
orderMCS_rGT_trimm_AE = dict()
orderMCS_kGT_expand_nAE = dict()
orderMCS_rGT_expand_nAE = dict()
orderMCS_kGT_trimm_nAE = dict()
orderMCS_rGT_trimm_nAE = dict()
if(makePlots["LineplotsMCS"]):
    # task message
    print("\n")
    print("* Making plots of order of MCS between primal and consensus graphs ...")
    # initialize boxplot holders
    for i in range(0, 7):
        # without ambiguous edges
        orderMCS_kGT_expand_AE[i] = []
        orderMCS_rGT_expand_AE[i] = []
        orderMCS_kGT_trimm_AE[i] = []
        orderMCS_rGT_trimm_AE[i] = []
        orderMCS_kGT_expand_nAE[i] = []
        orderMCS_rGT_expand_nAE[i] = []
        orderMCS_kGT_trimm_nAE[i] = []
        orderMCS_rGT_trimm_nAE[i] = []
        # cluster distance data
        distMCS_kGT_expand_AE[i] = []
        distMCS_rGT_expand_AE[i] = []
        distMCS_kGT_trimm_AE[i] = []
        distMCS_rGT_trimm_AE[i] = []
        distMCS_kGT_expand_nAE[i] = []
        distMCS_rGT_expand_nAE[i] = []
        distMCS_kGT_trimm_nAE[i] = []
        distMCS_rGT_trimm_nAE[i] = []
        # cluster consensus data
        consOrd_kGT_expand_AE[i] = []
        consOrd_rGT_expand_AE[i] = []
        consOrd_kGT_trimm_AE[i] = []
        consOrd_rGT_trimm_AE[i] = []
        consOrd_kGT_expand_nAE[i] = []
        consOrd_rGT_expand_nAE[i] = []
        consOrd_kGT_trimm_nAE[i] = []
        consOrd_rGT_trimm_nAE[i] = []
    # get data from files
    for eachFile in inputFiles:
        # open file
        inputFile = open(eachFile, "rb")
        inputTuple = pkl.load(inputFile)
        inputFile.close()
        # unpack dictionaries
        inputDict_kGT_expand_AE = inputTuple[0]
        inputDict_kGT_expand_nAE = inputTuple[1]
        inputDict_rGT_expand_AE = inputTuple[2]
        inputDict_rGT_expand_nAE = inputTuple[3]
        inputDict_kGT_trimm_AE = inputTuple[4]
        inputDict_kGT_trimm_nAE = inputTuple[5]
        inputDict_rGT_trimm_AE = inputTuple[6]
        inputDict_rGT_trimm_nAE = inputTuple[7]
        # get required data
        row_kGT_expand_AE = inputDict_kGT_expand_AE["PrimalConsensus"]
        row_kGT_expand_nAE = inputDict_kGT_expand_nAE["PrimalConsensus"]
        row_rGT_expand_AE = inputDict_rGT_expand_AE["PrimalConsensus"]
        row_rGT_expand_nAE = inputDict_rGT_expand_nAE["PrimalConsensus"]
        row_kGT_trimm_AE = inputDict_kGT_trimm_AE["PrimalConsensus"]
        row_kGT_trimm_nAE = inputDict_kGT_trimm_nAE["PrimalConsensus"]
        row_rGT_trimm_AE = inputDict_rGT_trimm_AE["PrimalConsensus"]
        row_rGT_trimm_nAE = inputDict_rGT_trimm_nAE["PrimalConsensus"]
        # get order of primal
        primal_kGT_expand_AE = inputDict_kGT_expand_AE["Primal"].order()
        primal_kGT_expand_nAE = inputDict_kGT_expand_nAE["Primal"].order()
        primal_rGT_expand_AE = inputDict_rGT_expand_AE["Primal"].order()
        primal_rGT_expand_nAE = inputDict_rGT_expand_nAE["Primal"].order()
        primal_kGT_trimm_AE = inputDict_kGT_trimm_AE["Primal"].order()
        primal_kGT_trimm_nAE = inputDict_kGT_trimm_nAE["Primal"].order()
        primal_rGT_trimm_AE = inputDict_rGT_trimm_AE["Primal"].order()
        primal_rGT_trimm_nAE = inputDict_rGT_trimm_nAE["Primal"].order()
        # get order of consensus
        Keys = [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7)]
        consensus_kGT_expand_AE = [inputDict_kGT_expand_AE["ConsensusGraphs"][(a, b)].order() for (a, b) in Keys]
        consensus_kGT_expand_nAE = [inputDict_kGT_expand_nAE["ConsensusGraphs"][(a, b)].order() for (a, b) in Keys]
        consensus_rGT_expand_AE = [inputDict_rGT_expand_AE["ConsensusGraphs"][(a, b)].order() for (a, b) in Keys]
        consensus_rGT_expand_nAE = [inputDict_rGT_expand_nAE["ConsensusGraphs"][(a, b)].order() for (a, b) in Keys]
        consensus_kGT_trimm_AE = [inputDict_kGT_trimm_AE["ConsensusGraphs"][(a, b)].order() for (a, b) in Keys]
        consensus_kGT_trimm_nAE = [inputDict_kGT_trimm_nAE["ConsensusGraphs"][(a, b)].order() for (a, b) in Keys]
        consensus_rGT_trimm_AE = [inputDict_rGT_trimm_AE["ConsensusGraphs"][(a, b)].order() for (a, b) in Keys]
        consensus_rGT_trimm_nAE = [inputDict_rGT_trimm_nAE["ConsensusGraphs"][(a, b)].order() for (a, b) in Keys]
        # get distance
        dist_kGT_expand_AE = [primal_kGT_expand_AE + consensus_kGT_expand_AE[i] - 2*row_kGT_expand_AE[i] for i in range(len(row_kGT_expand_AE))]
        dist_kGT_expand_nAE = [primal_kGT_expand_nAE + consensus_kGT_expand_nAE[i] - 2*row_kGT_expand_nAE[i] for i in range(len(row_kGT_expand_nAE))]
        dist_rGT_expand_AE = [primal_rGT_expand_AE + consensus_rGT_expand_AE[i] - 2*row_rGT_expand_AE[i] for i in range(len(row_rGT_expand_AE))]
        dist_rGT_expand_nAE = [primal_rGT_expand_nAE + consensus_rGT_expand_nAE[i] - 2*row_rGT_expand_nAE[i] for i in range(len(row_rGT_expand_nAE))]
        dist_kGT_trimm_AE = [primal_kGT_trimm_AE + consensus_kGT_trimm_AE[i] - 2*row_kGT_trimm_AE[i] for i in range(len(row_kGT_trimm_AE))]
        dist_kGT_trimm_nAE = [primal_kGT_trimm_nAE + consensus_kGT_trimm_nAE[i] - 2*row_kGT_trimm_nAE[i] for i in range(len(row_kGT_trimm_nAE))]
        dist_rGT_trimm_AE = [primal_rGT_trimm_AE + consensus_rGT_trimm_AE[i] - 2*row_rGT_trimm_AE[i] for i in range(len(row_rGT_trimm_AE))]
        dist_rGT_trimm_nAE = [primal_rGT_trimm_nAE + consensus_rGT_trimm_nAE[i] - 2*row_rGT_trimm_nAE[i] for i in range(len(row_rGT_trimm_nAE))]
        # save data accordingly
        for i in range(0, 7):
            # order of MCS
            orderMCS_kGT_expand_AE[i].append(row_kGT_expand_AE[i])
            orderMCS_rGT_expand_AE[i].append(row_rGT_expand_AE[i])
            orderMCS_kGT_trimm_AE[i].append(row_kGT_trimm_AE[i])
            orderMCS_rGT_trimm_AE[i].append(row_rGT_trimm_AE[i])
            orderMCS_kGT_expand_nAE[i].append(row_kGT_expand_nAE[i])
            orderMCS_rGT_expand_nAE[i].append(row_rGT_expand_nAE[i])
            orderMCS_kGT_trimm_nAE[i].append(row_kGT_trimm_nAE[i])
            orderMCS_rGT_trimm_nAE[i].append(row_rGT_trimm_nAE[i])
            # cluster distance data
            distMCS_kGT_expand_AE[i].append(dist_kGT_expand_AE[i])
            distMCS_rGT_expand_AE[i].append(dist_rGT_expand_AE[i])
            distMCS_kGT_trimm_AE[i].append(dist_kGT_trimm_AE[i])
            distMCS_rGT_trimm_AE[i].append(dist_rGT_trimm_AE[i])
            distMCS_kGT_expand_nAE[i].append(dist_kGT_expand_nAE[i])
            distMCS_rGT_expand_nAE[i].append(dist_rGT_expand_nAE[i])
            distMCS_kGT_trimm_nAE[i].append(dist_kGT_trimm_nAE[i])
            distMCS_rGT_trimm_nAE[i].append(dist_rGT_trimm_nAE[i])
            # cluster consensus data
            consOrd_kGT_expand_AE[i].append(consensus_kGT_expand_AE[i])
            consOrd_rGT_expand_AE[i].append(consensus_rGT_expand_AE[i])
            consOrd_kGT_trimm_AE[i].append(consensus_kGT_trimm_AE[i])
            consOrd_rGT_trimm_AE[i].append(consensus_rGT_trimm_AE[i])
            consOrd_kGT_expand_nAE[i].append(consensus_kGT_expand_nAE[i])
            consOrd_rGT_expand_nAE[i].append(consensus_rGT_expand_nAE[i])
            consOrd_kGT_trimm_nAE[i].append(consensus_kGT_trimm_nAE[i])
            consOrd_rGT_trimm_nAE[i].append(consensus_rGT_trimm_nAE[i])
    # average values
    for i in range(0, 7):
        # average order of MCS
        orderMCS_kGT_expand_AE_av.append(np.average(orderMCS_kGT_expand_AE[i]))
        orderMCS_rGT_expand_AE_av.append(np.average(orderMCS_rGT_expand_AE[i]))
        orderMCS_kGT_trimm_AE_av.append(np.average(orderMCS_kGT_trimm_AE[i]))
        orderMCS_rGT_trimm_AE_av.append(np.average(orderMCS_rGT_trimm_AE[i]))
        orderMCS_kGT_expand_nAE_av.append(np.average(orderMCS_kGT_expand_nAE[i]))
        orderMCS_rGT_expand_nAE_av.append(np.average(orderMCS_rGT_expand_nAE[i]))
        orderMCS_kGT_trimm_nAE_av.append(np.average(orderMCS_kGT_trimm_nAE[i]))
        orderMCS_rGT_trimm_nAE_av.append(np.average(orderMCS_rGT_trimm_nAE[i]))
        # average order of consensus
        consOrd_kGT_expand_AE_av.append(np.average(consOrd_kGT_expand_AE[i]))
        consOrd_rGT_expand_AE_av.append(np.average(consOrd_rGT_expand_AE[i]))
        consOrd_kGT_trimm_AE_av.append(np.average(consOrd_kGT_trimm_AE[i]))
        consOrd_rGT_trimm_AE_av.append(np.average(consOrd_rGT_trimm_AE[i]))
        consOrd_kGT_expand_nAE_av.append(np.average(consOrd_kGT_expand_nAE[i]))
        consOrd_rGT_expand_nAE_av.append(np.average(consOrd_rGT_expand_nAE[i]))
        consOrd_kGT_trimm_nAE_av.append(np.average(consOrd_kGT_trimm_nAE[i]))
        consOrd_rGT_trimm_nAE_av.append(np.average(consOrd_rGT_trimm_nAE[i]))
        # average of distance data
        distMCS_kGT_expand_AE_av.append(np.average(distMCS_kGT_expand_AE[i]))
        distMCS_rGT_expand_AE_av.append(np.average(distMCS_rGT_expand_AE[i]))
        distMCS_kGT_trimm_AE_av.append(np.average(distMCS_kGT_trimm_AE[i]))
        distMCS_rGT_trimm_AE_av.append(np.average(distMCS_rGT_trimm_AE[i]))
        distMCS_kGT_expand_nAE_av.append(np.average(distMCS_kGT_expand_nAE[i]))
        distMCS_rGT_expand_nAE_av.append(np.average(distMCS_rGT_expand_nAE[i]))
        distMCS_kGT_trimm_nAE_av.append(np.average(distMCS_kGT_trimm_nAE[i]))
        distMCS_rGT_trimm_nAE_av.append(np.average(distMCS_rGT_trimm_nAE[i]))

    print(distMCS_kGT_expand_AE_av)
    print(distMCS_kGT_expand_nAE_av)
    print("--------------------------")
    print(distMCS_kGT_trimm_AE_av)
    print(distMCS_kGT_trimm_nAE_av)    
        
    # WITHOUT AMBIGUOUS EDGES ------------------------------------------------------
    # data for plots
    X = [1, 2, 3, 4, 5, 6, 7]
    XL = ["1/7", "2/7", "3/7", "4/7", "5/7", "6/7", "7/7"]
    # plot expand and trimm without AE
    fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle(r"Average order of MCIS between $G_0$ and Consensus Graphs", fontsize = 14)#, weight = "light")
    ax1.plot(X, orderMCS_kGT_expand_nAE_av, linestyle = "-", color = "steelblue", linewidth = 2, marker = "s", markersize = 5, label = "kernel-based GT")
    ax1.plot(X, orderMCS_rGT_expand_nAE_av, linestyle = "--", color = "slateblue", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and no ambiguous edges")
    ax1.plot(X, orderMCS_rGT_expand_AE_av, linestyle = ":", color = "orchid", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and ambiguous edges")
    ax1.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    ax1.set_xticks([])
    ax1.set_xticks(X, labels = XL, fontsize = 11)#, weight = "light")
    ax1.set_yticks([])
    ax1.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], fontsize = 11)#, weight = "light")
    # ax1.legend(fontsize = 10)
    axp = ax1.twinx()
    axp.set_yticks([])
    # axp.set_ylabel("VF2_step", rotation = -90, fontsize = 12, weight = "light", labelpad = 16)
    ax2.plot(X, orderMCS_kGT_trimm_nAE_av, linestyle = "-", color = "darkgreen", linewidth = 2, marker = "s", markersize = 5, label = "kernel-based GT")
    ax2.plot(X, orderMCS_rGT_trimm_nAE_av, linestyle = "--", color = "tab:orange", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and no ambiguous edges")
    ax2.plot(X, orderMCS_rGT_trimm_AE_av, linestyle = ":", color = "tab:red", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and ambiguous edges")
    ax2.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    ax2.set_xticks([])
    ax2.set_xticks(X, labels = XL, fontsize = 11)#, weight = "light")
    ax2.set_yticks([])
    ax2.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], fontsize = 11)#, weight = "light")
    # ax2.legend(fontsize = 10)
    axp = ax2.twinx()
    axp.set_yticks([])
    # axp.set_ylabel("Iterative_Trimming", rotation = -90, fontsize = 12, weight = "light", labelpad = 16)
    ax2.set_xlabel(" ", fontsize = 15)#, weight = "light")
    fig.supylabel("Average order", fontsize = 15)#, weight = "light")
    plt.tight_layout()
    plt.savefig(namePrimalVSConsensus)
    plt.close()

    # plot expand and trimm without AE
    fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle(r"Average distance between $G_0$ and Consensus Graphs", fontsize = 14)#, weight = "light")    
    ax1.plot(X, distMCS_kGT_expand_nAE_av, linestyle = "-", color = "steelblue", linewidth = 2, marker = "s", markersize = 5, label = "kernel-based GT")
    ax1.plot(X, distMCS_rGT_expand_nAE_av, linestyle = "--", color = "slateblue", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and no ambiguous edges")
    ax1.plot(X, distMCS_rGT_expand_AE_av, linestyle = ":", color = "orchid", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and ambiguoues edges")
    ax1.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    ax1.set_xticks([])
    ax1.set_xticks(X, labels = XL, fontsize = 11)#, weight = "light")
    # ax1.set_yticks([])
    # ax1.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], weight = "light", fontsize = 8)
    # ax1.legend(fontsize = 10)
    axp = ax1.twinx()
    axp.set_yticks([])
    # axp.set_ylabel("VF2_step", rotation = -90, fontsize = 12, weight = "light", labelpad = 16)
    ax2.plot(X, distMCS_kGT_trimm_nAE_av, linestyle = "-", color = "darkgreen", linewidth = 2, marker = "s", markersize = 5, label = "kernel-based GT")
    ax2.plot(X, distMCS_rGT_trimm_nAE_av, linestyle = "--", color = "tab:orange", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and no ambiguoues edges")
    ax2.plot(X, distMCS_rGT_trimm_AE_av, linestyle = ":", color = "tab:red", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and ambiguous edges")
    ax2.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    ax2.set_xticks([])
    ax2.set_xticks(X, labels = XL, fontsize = 11)#, weight = "light")
    # ax2.set_yticks([])
    # ax2.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], weight = "light", fontsize = 8)
    # ax2.legend(fontsize = 10)
    axp = ax2.twinx()
    axp.set_yticks([])
    # axp.set_ylabel("Iterative_Trimming", rotation = -90, fontsize = 12, weight = "light", labelpad = 16)
    ax2.set_xlabel(" ", fontsize = 15)#, weight = "light")
    fig.supylabel("MCIS-Distance", fontsize = 15)#, weight = "light")
    plt.tight_layout()
    plt.savefig(namePrimalVSConsensus_Dist)
    plt.close()

    # plot expand and trimm without AE
    fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle("Average Order of Consensus Graphs", fontsize = 14)#, weight = "light")    
    ax1.plot(X, consOrd_kGT_expand_nAE_av, linestyle = "-", color = "steelblue", linewidth = 2, marker = "s", markersize = 5, label = "kernel-based GT")
    ax1.plot(X, consOrd_rGT_expand_nAE_av, linestyle = "--", color = "slateblue", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and no ambiguous edges")
    ax1.plot(X, consOrd_rGT_expand_AE_av, linestyle = ":", color = "orchid", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and ambiguous edges")
    ax1.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    ax1.set_xticks([])
    ax1.set_xticks(X, labels = XL, fontsize = 11)#, weight = "light")
    # ax1.set_yticks([])
    # ax1.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], weight = "light", fontsize = 8)
    # ax1.legend(fontsize = 10)
    axp = ax1.twinx()
    axp.set_yticks([])
    axp.set_ylabel("VF2_step", rotation = -90, fontsize = 12, labelpad = 16)#, weight = "light")
    ax2.plot(X, consOrd_kGT_trimm_nAE_av, linestyle = "-", color = "darkgreen", linewidth = 2, marker = "s", markersize = 5, label = "kernel-based GT")
    ax2.plot(X, consOrd_rGT_trimm_nAE_av, linestyle = "--", color = "tab:orange", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and no ambiguous edges")
    ax2.plot(X, consOrd_rGT_trimm_AE_av, linestyle = ":", color = "tab:red", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT and ambiguous edges")
    ax2.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    ax2.set_xticks([])
    ax2.set_xticks(X, labels = XL, fontsize = 11)#, weight = "light")
    # ax2.set_yticks([])
    # ax2.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], weight = "light", fontsize = 8)
    # ax2.legend(fontsize = 10)
    axp = ax2.twinx()
    axp.set_yticks([])
    axp.set_ylabel("Iterative_Trimming", rotation = -90, fontsize = 12, labelpad = 16)#, weight = "light")
    ax2.set_xlabel(r"Voting threshold $\alpha$", fontsize = 15)#, weight = "light")
    fig.supylabel("Average order", fontsize = 15)#, weight = "light")
    plt.tight_layout()
    plt.savefig(namePrimalVSConsensus_Ord)
    plt.close()

    # # WITH AMBIGUOUS EDGES ---------------------------------------------------------
    # # data for plots
    # X = [1, 2, 3, 4, 5, 6, 7]
    # XL = ["1/7", "2/7", "3/7", "4/7", "5/7", "6/7", "7/7"]
    # # plot expand and trimm without AE
    # fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle(r"Average order of MCIS between $G_0$ and Consensus Graphs," + "\nconsidering  Ambiguous Edges", fontsize = 14, weight = "light")
    # ax1.plot(X, orderMCS_kGT_expand_AE_av, linestyle = "-", color = "steelblue", linewidth = 1.15, marker = "s", markersize = 4, label = "kernel-based GT")
    # ax1.plot(X, orderMCS_rGT_expand_AE_av, linestyle = "--", color = "cornflowerblue", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT")
    # ax1.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    # ax1.set_xticks([])
    # ax1.set_xticks(X, labels = XL, weight = "light", fontsize = 11)
    # ax1.set_yticks([])
    # ax1.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], weight = "light", fontsize = 11)
    # # ax1.legend(fontsize = 10)
    # axp = ax1.twinx()
    # axp.set_yticks([])
    # # axp.set_ylabel("VF2_step", rotation = -90, fontsize = 12, weight = "light", labelpad = 16)
    # ax2.plot(X, orderMCS_kGT_trimm_AE_av, linestyle = "-", color = "darkgreen", linewidth = 1.15, marker = "s", markersize = 4, label = "kernel-based GT")
    # ax2.plot(X, orderMCS_rGT_trimm_AE_av, linestyle = "--", color = "tab:green", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT")
    # ax2.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    # ax2.set_xticks([])
    # ax2.set_xticks(X, labels = XL, weight = "light", fontsize = 11)
    # ax2.set_yticks([])
    # ax2.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], weight = "light", fontsize = 11)
    # # ax2.legend(fontsize = 10)
    # axp = ax2.twinx()
    # axp.set_yticks([])
    # # axp.set_ylabel("Iterative_Trimming", rotation = -90, fontsize = 12, weight = "light", labelpad = 16)
    # ax2.set_xlabel(r"Voting threshold $\alpha$", weight = "light", fontsize = 15)
    # fig.supylabel("Average order", weight = "light", fontsize = 15)
    # plt.tight_layout()
    # plt.savefig(namePrimalVSConsensus_AE)
    # plt.close()

    # # plot expand and trimm without AE
    # fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle(r"Average distance between $G_0$ and Consensus Graphs," + "\nconsidering  Ambiguous Edges", fontsize = 14, weight = "light")
    # ax1.plot(X, distMCS_kGT_expand_AE_av, linestyle = "-", color = "steelblue", linewidth = 1.15, marker = "s", markersize = 4, label = "kernel-based GT")
    # ax1.plot(X, distMCS_rGT_expand_AE_av, linestyle = "--", color = "cornflowerblue", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT")
    # ax1.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    # ax1.set_xticks([])
    # ax1.set_xticks(X, labels = XL, weight = "light", fontsize = 11)
    # # ax1.set_yticks([])
    # # ax1.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], weight = "light", fontsize = 8)
    # # ax1.legend(fontsize = 10)
    # axp = ax1.twinx()
    # axp.set_yticks([])
    # # axp.set_ylabel("VF2_step", rotation = -90, fontsize = 12, weight = "light", labelpad = 16)
    # ax2.plot(X, distMCS_kGT_trimm_AE_av, linestyle = "-", color = "darkgreen", linewidth = 1.15, marker = "s", markersize = 4, label = "kernel-based GT")
    # ax2.plot(X, distMCS_rGT_trimm_AE_av, linestyle = "--", color = "tab:green", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT")
    # ax2.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    # ax2.set_xticks([])
    # ax2.set_xticks(X, labels = XL, weight = "light", fontsize = 11)
    # # ax2.set_yticks([])
    # # ax2.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], weight = "light", fontsize = 8)
    # # ax2.legend(fontsize = 10)
    # axp = ax2.twinx()
    # axp.set_yticks([])
    # # axp.set_ylabel("Iterative_Trimming", rotation = -90, fontsize = 12, weight = "light", labelpad = 16)
    # ax2.set_xlabel(r"Voting threshold $\alpha$", weight = "light", fontsize = 15)
    # fig.supylabel("MCIS-Distance", weight = "light", fontsize = 15)
    # plt.tight_layout()
    # plt.savefig(namePrimalVSConsensus_Dist_AE)
    # plt.close()

    # # plot expand and trimm without AE
    # fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle("Average Order of Consensus Graphs," + "\nconsidering  Ambiguous Edges", fontsize = 14, weight = "light")
    # ax1.plot(X, consOrd_kGT_expand_AE_av, linestyle = "-", color = "steelblue", linewidth = 1.15, marker = "s", markersize = 4, label = "kernel-based GT")
    # ax1.plot(X, consOrd_rGT_expand_AE_av, linestyle = "--", color = "cornflowerblue", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT")
    # ax1.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    # ax1.set_xticks([])
    # ax1.set_xticks(X, labels = XL, weight = "light", fontsize = 11)
    # # ax1.set_yticks([])
    # # ax1.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], weight = "light", fontsize = 8)
    # ax1.legend(fontsize = 12)
    # axp = ax1.twinx()
    # axp.set_yticks([])
    # axp.set_ylabel("VF2_step", rotation = -90, fontsize = 12, weight = "light", labelpad = 16)
    # ax2.plot(X, consOrd_kGT_trimm_AE_av, linestyle = "-", color = "darkgreen", linewidth = 1.15, marker = "s", markersize = 4, label = "kernel-based GT")
    # ax2.plot(X, consOrd_rGT_trimm_AE_av, linestyle = "--", color = "tab:green", linewidth = 1.15, marker = ".", markersize = 5, label = "random GT")
    # ax2.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    # ax2.set_xticks([])
    # ax2.set_xticks(X, labels = XL, weight = "light", fontsize = 11)
    # # ax2.set_yticks([])
    # # ax2.set_yticks([8, 10, 12, 14], labels = [8, 10, 12, 14], weight = "light", fontsize = 8)
    # ax2.legend(fontsize = 12)
    # axp = ax2.twinx()
    # axp.set_yticks([])
    # axp.set_ylabel("Iterative_Trimming", rotation = -90, fontsize = 12, weight = "light", labelpad = 16)
    # ax2.set_xlabel(r"Voting threshold $\alpha$", weight = "light", fontsize = 15)
    # fig.supylabel("Average order", weight = "light", fontsize = 15)
    # plt.tight_layout()
    # plt.savefig(namePrimalVSConsensus_Ord_AE)
    # plt.close()





# scatterplot: number of gaos vs number of ambiguous edges ---------------------
if(makePlots["ScatterGaps"]):
    # task message
    print("\n")
    print("* Making running time boxplots ...")
    # get data from files
    for eachFile in inputFiles:
        # open file
        inputFile = open(eachFile, "rb")
        inputTuple = pkl.load(inputFile)
        inputFile.close()
        # unpack dictionaries
        inputDict_kGT_expand_AE = inputTuple[0]
        inputDict_rGT_expand_AE = inputTuple[2]
        inputDict_kGT_trimm_AE = inputTuple[4]
        inputDict_rGT_trimm_AE = inputTuple[6]
        inputDict_kGT_expand_nAE = inputTuple[1]
        inputDict_rGT_expand_nAE = inputTuple[3]
        inputDict_kGT_trimm_nAE = inputTuple[5]
        inputDict_rGT_trimm_nAE = inputTuple[7]
        # get required data
        gaps_inputDict_kGT_expand_AE.append(inputDict_kGT_expand_AE["GapsAndAEs"][0])
        ambs_inputDict_kGT_expand_AE.append(inputDict_kGT_expand_AE["GapsAndAEs"][1])
        gaps_inputDict_rGT_expand_AE.append(inputDict_rGT_expand_AE["GapsAndAEs"][0])
        ambs_inputDict_rGT_expand_AE.append(inputDict_rGT_expand_AE["GapsAndAEs"][1])
        gaps_inputDict_kGT_trimm_AE.append(inputDict_kGT_trimm_AE["GapsAndAEs"][0])
        ambs_inputDict_kGT_trimm_AE.append(inputDict_kGT_trimm_AE["GapsAndAEs"][1])
        gaps_inputDict_rGT_trimm_AE.append(inputDict_rGT_trimm_AE["GapsAndAEs"][0])
        ambs_inputDict_rGT_trimm_AE.append(inputDict_rGT_trimm_AE["GapsAndAEs"][1])
        gaps_inputDict_kGT_expand_nAE.append(inputDict_kGT_expand_nAE["GapsAndAEs"][0])
        ambs_inputDict_kGT_expand_nAE.append(inputDict_kGT_expand_nAE["GapsAndAEs"][1])
        gaps_inputDict_rGT_expand_nAE.append(inputDict_rGT_expand_nAE["GapsAndAEs"][0])
        ambs_inputDict_rGT_expand_nAE.append(inputDict_rGT_expand_nAE["GapsAndAEs"][1])
        gaps_inputDict_kGT_trimm_nAE.append(inputDict_kGT_trimm_nAE["GapsAndAEs"][0])
        ambs_inputDict_kGT_trimm_nAE.append(inputDict_kGT_trimm_nAE["GapsAndAEs"][1])
        gaps_inputDict_rGT_trimm_nAE.append(inputDict_rGT_trimm_nAE["GapsAndAEs"][0])
        ambs_inputDict_rGT_trimm_nAE.append(inputDict_rGT_trimm_nAE["GapsAndAEs"][1])
    # get correlation
    GAPS = gaps_inputDict_kGT_expand_AE + gaps_inputDict_rGT_expand_AE + gaps_inputDict_kGT_trimm_AE + gaps_inputDict_rGT_trimm_AE + gaps_inputDict_kGT_expand_nAE + gaps_inputDict_rGT_expand_nAE + gaps_inputDict_kGT_trimm_nAE + gaps_inputDict_rGT_trimm_nAE
    AMBS = ambs_inputDict_kGT_expand_AE + ambs_inputDict_rGT_expand_AE + ambs_inputDict_kGT_trimm_AE + ambs_inputDict_rGT_trimm_AE + ambs_inputDict_kGT_expand_nAE + ambs_inputDict_rGT_expand_nAE + ambs_inputDict_kGT_trimm_nAE + ambs_inputDict_rGT_trimm_nAE
    linReg = sp.stats.linregress(GAPS, AMBS)
    minG = min(GAPS)
    maxG = max(GAPS)
    minA = min(AMBS)
    maxA = max(AMBS)
    X = list(range(45, 115+1))
    Y = [(linReg.slope)*x + linReg.intercept for x in X]
    # make scatter plots
    fig, ax = plt.subplots()
    theLegend = "Pearson correlation\n" + r"$\rho$ = " + str(round(linReg.rvalue, 5))# + "\nm = " + str(round(linReg.slope, 5)) + "\nb = " + str(round(linReg.intercept, 5))
    plt.plot(X, Y, linestyle = "--", color = "tab:red", linewidth = 1.5, label = theLegend)
    plt.scatter(GAPS, AMBS)
    plt.xlabel("Number of gaps", fontsize = 14)#, weight = "light")
    plt.ylabel("Number of ambiguous edges", fontsize = 14)#, weight = "light")
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    ax.set_axisbelow(True)
    plt.grid(color = "lightgray", linestyle = "--", linewidth = 0.6)
    plt.legend(fontsize = 10)
    # plt.title("Linear regression: Gaps vs Ambiguous Edges in Aignments", fontsize = 14, weight = "light")
    plt.xlim(45, 115)
    plt.tight_layout()
    plt.savefig(nameScatterGapsAmbs)
    plt.close()
    # plot histogram of ambiguous edges in - expand - with AE - kernelGT
    fig, ax = plt.subplots()
    mean_ae_expand = mean(ambs_inputDict_kGT_expand_nAE)
    plt.hist(ambs_inputDict_kGT_expand_nAE, bins = 22, rwidth = 0.98, alpha = 0.9, color = "steelblue")
    plt.vlines(mean_ae_expand, color = "tab:red", ymin = 0, ymax = 12.5,
               linewidth = 2, linestyle = "--",
               label = "mean: " + str(round(mean_ae_expand, 2)) + " ambiguous edges")
    plt.legend(fontsize = 11)
    plt.ylim(0, 7.5)
    plt.xlabel("Number of Ambiguous Edges in Alignments\n obtained with kernel-based GT", fontsize = 14)#, weight = "light")
    plt.ylabel("Number of scenarios", fontsize = 14)#, weight = "light")
    plt.xlim(xmin = 20, xmax = 150)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    ax.set_axisbelow(True)
    plt.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    plt.tight_layout()
    plt.savefig(nameHistGapsKernelGT)
    plt.close()
    # plot histogram of ambiguous edges in - expand - with AE - randomGT
    fig, ax = plt.subplots()
    mean_ae_expand = mean(ambs_inputDict_rGT_expand_nAE)
    plt.hist(ambs_inputDict_rGT_expand_nAE, bins = 22, rwidth = 0.98, alpha = 0.9, color = "cornflowerblue")
    plt.vlines(mean_ae_expand, color = "tab:red", ymin = 0, ymax = 12.5,
               linewidth = 2, linestyle = "--",
               label = "mean: " + str(round(mean_ae_expand, 2)) + " ambiguous edges")
    plt.legend(fontsize = 11)
    plt.ylim(0, 7.5)
    plt.xlabel("Number of Ambiguous Edges in Alignments\n obtained with random-based GT", fontsize = 14)#, weight = "light")
    plt.ylabel("Number of scenarios", fontsize = 14)#, weight = "light")
    plt.xlim(xmin = 20, xmax = 150)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    ax.set_axisbelow(True)
    plt.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    plt.tight_layout()
    plt.savefig(nameHistGapsRandomGT)
    plt.close()




# scatterplot: number of gaos vs number of ambiguous edges ---------------------
if(makePlots["ScatterDist"]):
    # get data from files
    for eachFile in inputFiles:
        # open file
        inputFile = open(eachFile, "rb")
        inputTuple = pkl.load(inputFile)
        inputFile.close()
        # get dictionary
        theDict = inputTuple[0]
        # get primal-mutant kernel distance
        pmValsK = pmValsK + theDict["KernelDistancePMVals"]
        # get primal-mutant score distance
        pmValsS = pmValsS + theDict["ScoreDistancePMVals"]
        # get mutant-mutant kernel distance
        mmValsK = mmValsK + theDict["KernelDistanceMMVals"]
        # get mutant-mutant score distance
        mmValsS = mmValsS + theDict["ScoreDistanceMMVals"]

    # task message
    pmValsK = [sqrt(eachVal) for eachVal in pmValsK]
    # pmValsS = [sqrt(eachVal) for eachVal in pmValsS]
    linReg = sp.stats.linregress(pmValsK, pmValsS)
    minX = min(pmValsK)
    maxX = max(pmValsK)
    X = [0, 0.09]
    Y = [(linReg.slope)*x + linReg.intercept for x in X]
    theLegend = "Pearson correlation\n" + r"$\rho$ = " + str(round(linReg.rvalue, 5))# + "\nm = " + str(round(linReg.slope, 5)) + "\nb = " + str(round(linReg.intercept, 5))
    print("\n")
    print("*** Making scatter plot of score-distance vs kernel-distance between primal and mutants ...")
    fig, ax = plt.subplots()
    plt.scatter(pmValsK, pmValsS)
    plt.plot(X, Y, linestyle = "--", color = "tab:red", linewidth = 2, label = theLegend)
    plt.title(r"Linear Regression: Distance between $G_0$ and its Mutants", fontsize = 14)#, weight = "light")
    plt.xlabel("Kernel-based Distance", fontsize = 14)#, weight = "light")
    plt.ylabel("MCIS-based Distance", fontsize = 14)#, weight = "light")
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlim(0.01, 0.09)
    plt.ylim(1, 15)
    plt.legend(fontsize = 10)
    ax.set_axisbelow(True)
    plt.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    plt.tight_layout()
    plt.savefig(nameScatterPM)
    plt.close()

    # task message
    mmValsK = [sqrt(eachVal) for eachVal in mmValsK]
    # mmValsS = [sqrt(eachVal) for eachVal in mmValsS]
    linReg = sp.stats.linregress(mmValsK, mmValsS)
    minX = min(mmValsK)
    maxX = max(mmValsK)
    X = [0.01, 0.09]
    Y = [(linReg.slope)*x + linReg.intercept for x in X]
    theLegend = "Pearson correlation\n" + r"$\rho$ = " + str(round(linReg.rvalue, 5))# + "\nm = " + str(round(linReg.slope, 5)) + "\nb = " + str(round(linReg.intercept, 5))
    print("\n")
    print("*** Making scatter plot of score-distance vs kernel-distance between pairs of mutants ...")
    fig, ax = plt.subplots()
    plt.scatter(mmValsK, mmValsS)
    plt.plot(X, Y, linestyle = "--", color = "tab:red", linewidth = 2, label = theLegend)
    plt.title("Linear Regression: Distance between Pairs of Mutants", fontsize = 14)#, weight = "light")
    plt.xlabel("Kernel-based Distance", fontsize = 14)#, weight = "light")
    plt.ylabel("MCIS-based Distance", fontsize = 14)#, weight = "light")
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.ylim(1, 15)
    plt.xlim(0.01, 0.09)
    plt.legend(fontsize = 10)
    ax.set_axisbelow(True)
    plt.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    plt.tight_layout()
    plt.savefig(nameScatterMM)
    plt.close()




# histogram: order of MCS betwen primals and mutants ---------------------------
if(makePlots["HistogramDist"]):
    # get MCS values
    mcsValsPM = []
    mcsValsMM = []
    # get data from files
    for eachFile in inputFiles:
        # open file
        inputFile = open(eachFile, "rb")
        inputTuple = pkl.load(inputFile)
        inputFile.close()
        # get dictionary
        theDict = inputTuple[0]
        # get primal-mutant score distance
        mcsValsPM = mcsValsPM + [theDict["ScoreDistancePM"][eachKey][2] for eachKey in list(theDict["ScoreDistancePM"].keys())]
        # get mutant-mutant score distance
        mcsValsMM = mcsValsMM + [theDict["ScoreDistanceMM"][eachKey][2] for eachKey in list(theDict["ScoreDistanceMM"].keys())]

    # task message
    fig, ax = plt.subplots()
    mean_mcs = mean(mcsValsPM)
    plt.hist(mcsValsPM, bins = [10, 11, 12, 13, 14, 15, 16], rwidth = 0.98, alpha = 0.5,
             color = "steelblue", align = "left")
    plt.vlines(mean_mcs, color = "tab:red", ymin = 0, ymax = 100, linewidth = 2, linestyle = "--",
               label = "mean: " + str(round(mean_mcs, 2)) + " vertices")
    plt.legend(fontsize = 11)
    plt.ylim(0, 100)
    plt.xlim(7, 17)
    plt.xlabel(r"Order of MCIS between $G_0$ and mutants", fontsize = 14)#, weight = "light")
    plt.ylabel("Number of pairwise comparisons", fontsize = 14)#, weight = "light")
    plt.xticks([8, 9, 10, 11, 12, 13, 14, 15, 16], [8, 9, 10, 11, 12, 13, 14, 15, 16], fontsize = 10)
    ax.set_axisbelow(True)
    plt.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    plt.tight_layout()
    plt.savefig(nameHistMCS_PM)
    plt.close()

    # task message
    fig, ax = plt.subplots()
    mean_mcs = mean(mcsValsMM)
    plt.hist(mcsValsMM, bins = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17], rwidth = 0.98, alpha = 0.5,
            color = "cornflowerblue", align = "left")
    plt.vlines(mean_mcs, color = "tab:red", ymin = 0, ymax = 250, linewidth = 2, linestyle = "--",
               label = "mean: " + str(round(mean_mcs, 2)) + " vertices")
    plt.legend(fontsize = 11)
    plt.ylim(0, 250)
    plt.xlabel(r"Order of MCIS between pairs of mutants", fontsize = 14)#, weight = "light")
    plt.ylabel("Number of pairwise comparisons", fontsize = 14)#, weight = "light")
    plt.xticks([], fontsize = 12)
    plt.xticks([8, 9, 10, 11, 12, 13, 14, 15, 16], [8, 9, 10, 11, 12, 13, 14, 15, 16], fontsize = 10)
    ax.set_axisbelow(True)
    plt.grid(color = "lightgray", linestyle = "--", linewidth = 0.7)
    plt.tight_layout()
    plt.savefig(nameHistMCS_MM)
    plt.close()


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
