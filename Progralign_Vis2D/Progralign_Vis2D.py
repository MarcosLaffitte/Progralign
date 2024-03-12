################################################################################
#                                                                              #
#  README - Program: Progralign_Vis2D.py                                       #
#                                                                              #
#  - Paper: https://www.mdpi.com/1999-4893/17/3/116                            #
#                                                                              #
#  - Github repository: https://github.com/MarcosLaffitte/Progralign           #
#                                                                              #
#  - Date: 12 March 2024                                                       #
#                                                                              #
#  - Contributor(s) to this script:                                            #
#    * @MarcosLaffitte - Marcos E. Gonzalez Laffitte                           #
#                                                                              #
#  - Description: receives the output of Progralign_Analyisis and produces     #
#    two pdf files with different visualizations of the alignment and the      #
#    input graphs. If only 5 input graphs were aligned then it also produces   #
#    an especial file containig data to make a 3D plot with Progralign_Vis3D.  #
#                                                                              #
#  - NOTE: the position of the vertices in the alignment, and hence in the     #
#    input graphs, can be set by hand in-code, otherwise this are produced     #
#    by the layout packages of NetworkX. See DEFINE comment below.             #
#                                                                              #
#  - Input: a pickled python dictionary with the results of the alignment      #
#    obtained by Progralign_Analysis.                                          #
#                                                                              #
#  - Run with (after activating [pgalign] conda environment):                  #
#                                                                              #
#              python  Progralign_Vis2D.py  example_Results.pkl                #
#                                                                              #
#  - Output:                                                                   #
#    (a) example_Results_1D_plot.pdf                                           #
#    (b) example_Results_2D_plot.pdf                                           #
#    (c) example_Results_3D_data.pkl (for at most 5 graphs, set in-code)       #
#                                                                              #
#  NOTE: the 3D plots made by Progralign_Vis3D are made by stacking the 2D     #
#  plots that are the output of Progralign_Vis2D, and this cannot be changed   #
#  later. The position of the vertices can be entered in-code if required.     #
#                                                                              #
#  --------------------------------------------------------------------------  #
#                                                                              #
# - LICENSE:                                                                   #
#                                                                              #
#   This file is part of the work published in                                 #
#            https://www.mdpi.com/1999-4893/17/3/116                           #
#   and it is released under                                                   #
#            MIT License Copyright (c) 2023 Marcos E. González Laffitte        #
#   See LICENSE file in                                                        #
#            https://github.com/MarcosLaffitte/Progralign                      #
#   for full license details.                                                  #
#                                                                              #
################################################################################


# requires / used versions #####################################################
"""
> Lenguaje: python 3.10.12
> Anaconda: conda 23.7.3
> Packages installed with anaconda:
***** numpy 1.25.2
***** networkx 2.8.4
***** matplotlib 3.10.12
> Packages already in python:
***** pickle 4.0
"""


# dependencies #################################################################


# installed with conda ---------------------------------------------------------
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# already in python ------------------------------------------------------------
import time
import random
import pickle as pkl
from sys import argv
from copy import deepcopy
from math import modf, sqrt, ceil
from operator import eq, itemgetter
from itertools import product, combinations


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# input and variables ##########################################################


# user input for drawing and functionalities -----------------------------------
upperBound3D = 5
upperBound2D = 50
upperBound1D = 100


# input ------------------------------------------------------------------------
inputFileName = ""
inputFileName = argv[1]
inputList = []
inputFile = None


# data holders -----------------------------------------------------------------
dendrogramSSP = ""
indices = []
allLeaves = []
graphFigs3DInfo = []
existence = dict()
mapToRoot = dict()
allLeavesGraphs = dict()
constructionDict = dict()
vis1D = False
vis2D = False
vis3D = False
finalAlignment = None


# output -----------------------------------------------------------------------
outputFile = None
name1DPlotsPDF = inputFileName.replace(".pkl", "_1D_plot.pdf")
name2DPlotsPDF = inputFileName.replace(".pkl", "_2D_plot.pdf")
name3DPlotsPKL = inputFileName.replace(".pkl", "_3D_data.pkl")


# main #########################################################################


# initial message
print("\n")
print(">>> Progralign 2D Visualization - Progralign Github Repository")


# task message
print("\n")
print("* Retrieving input file ...")


# retrieve input file
inputFile = open(inputFileName, "rb")
constructionDict = pkl.load(inputFile)
inputFile.close()


# task message
print("\n")
print("*** Making plots ...")


# data for plots and evaluation of main alignment
vis1D = False
vis2D = False
vis3D = False
graphFigs3DInfo = []
indices = deepcopy(constructionDict["Indices"])
allLeaves = deepcopy(constructionDict["AllLeaves"])
existence = deepcopy(constructionDict["Existence"])
allLeavesGraphs = deepcopy(constructionDict["AllLeavesGraphs"])
mapToRoot = deepcopy(constructionDict["MapToRoot"])
dendrogramSSP = deepcopy(constructionDict["Dendrogram"])


# check maximum number of graphs
if(len(indices) <= upperBound3D):
    outText = "*** Received " + str(len(indices)) + " graphs. Saving 3D information and making 2D and 1D plots ..."
    print(outText)
    vis1D = True
    vis2D = True
    vis3D = True
if(upperBound3D < len(indices) <= upperBound2D):
    outText = "*** Received " + str(len(indices)) + " graphs.\n"
    outText = outText + "*** The creation of 3D plots is only possible with at most " + str(upperBound3D) + " graphs.\n"
    outText = outText + "*** Making only 2D and 1D plots ..."
    print(outText)
    vis1D = True
    vis2D = True
if(upperBound2D < len(indices) <= upperBound1D):
    outText = "*** Received " + str(len(indices)) + " graphs.\n"
    outText = outText + "*** The creation of 3D and 2D plots is only possible with at most "
    outText = outText + str(upperBound3D) + " and " + str(upperBound2D) + " graphs, respectively.\n"
    outText = outText + "*** Making only 1D plot ..."
    print(outText)
    vis1D = True
if(len(indices) > upperBound1D):
    errorStr = "*** Received " + str(len(indices)) + " graphs. Cannot make drawings,\n"
    errorStr = errorStr + "*** this is only possible with at most " + str(upperBound1D) + " graphs.\n"
    print(errorStr)


# prepare drawing data and attributes
if(vis1D or vis2D or vis3D):
    # get data
    finalAlignment = deepcopy(constructionDict["Alignment"])
    # drawing parameters
    minX = -1.15
    maxX = 1.15
    minY = -1.15
    maxY = 1.15
    apart = 0.1
    # figure-size offset and parameters
    wOffset = 7
    hOffset = 5
    nodesPerInch = 5
    graphsPerInch = 5
    nodesAlignment = list(finalAlignment.nodes())
    nodesAlignment.sort()
    vExtra = max(0, ceil((len(nodesAlignment)-(wOffset*nodesPerInch))/nodesPerInch))
    gExtra = max(0, ceil((len(indices)-(hOffset*graphsPerInch))/graphsPerInch))
    # get positions for vertices in the alignment
    fan = finalAlignment.order()
    valK = (1/sqrt(fan)) + apart
    # *** DEFINE here custom positions for the alignment nodes
    # *** as pairs (a, b) with a and b in the interval [-1, 1],
    # *** e.g. posAlignment[0] = (0.75, -1)
    # *** and comment the following line
    posAlignment = nx.spring_layout(finalAlignment, center = [0, 0], k = valK)
    # get color code of vertex labels
    k = 0
    nodeInfoTuple = []
    labelColor = dict()
    for eachLeaf in allLeaves:
        graphLeaf = deepcopy(allLeavesGraphs[eachLeaf])
        for (v, nodeInfo) in list(graphLeaf.nodes(data = True)):
            nodeInfoTuple = [(labelName, nodeInfo["nodeLabel"][labelName]) for labelName in list(nodeInfo["nodeLabel"].keys())]
            nodeInfoTuple.sort()
            nodeInfoTuple = tuple(nodeInfoTuple)
            if(not nodeInfoTuple in list(labelColor.keys())):
                k = k + 1
                labelColor[nodeInfoTuple] = k
    # set color map based on values
    allValues = list(labelColor.values())
    if(len(allValues) == 1):
        myCmap = plt.cm.Blues
    else:
        myCmap = plt.cm.turbo
    myCmap.set_bad("white", 1.)
    # add space on border of colormap and map values
    minValue = min(allValues)-0.8
    maxValue = max(allValues)+0.8
    allValues = allValues + [minValue, maxValue]
    allValues.sort()
    myNorm = plt.Normalize(vmin = minValue, vmax = maxValue)
    colorCode = myCmap(myNorm(allValues))
    # save alignmet 3D information
    if(vis3D):
        tupleInfo3D = []
        tupleInfo3D.append(finalAlignment.order())
        tupleInfo3D.append("Alignment")
        tupleInfo3D.append(deepcopy(posAlignment))
        tupleInfo3D.append(list(finalAlignment.nodes()))
        tupleInfo3D.append("silver")
        tupleInfo3D.append(deepcopy(finalAlignment))
        tupleInfo3D.append({v:v for v in list(finalAlignment.nodes())})
        graphFigs3DInfo.append(deepcopy(tuple(tupleInfo3D)))


# 2D visualization (and save 3D data if feasible)
if(vis2D):
    # create file
    sanityCheckPDF = PdfPages(name2DPlotsPDF)
    # draw input graphs
    for eachLeaf in allLeaves:
        # get position for vertices relative to alignment
        graphLeaf = deepcopy(allLeavesGraphs[eachLeaf])
        posLeaf = {v:posAlignment[mapToRoot[eachLeaf][v]] for v in list(graphLeaf.nodes())}
        # get color for vertices
        nodeList = []
        nodeColor = []
        for (v, nodeInfo) in list(graphLeaf.nodes(data = True)):
            nodeInfoTuple = [(labelName, nodeInfo["nodeLabel"][labelName]) for labelName in list(nodeInfo["nodeLabel"].keys())]
            nodeInfoTuple.sort()
            nodeInfoTuple = tuple(nodeInfoTuple)
            nodeColor.append(colorCode[labelColor[nodeInfoTuple]])
            nodeList.append(v)
        # create figure
        fig = plt.figure(figsize = [wOffset + vExtra, hOffset + vExtra])
        # plot alignment in background
        nx.draw_networkx(finalAlignment, with_labels = False, pos = posAlignment, node_size = 15,
                         node_color = "lightgrey", edge_color = "lightgrey", width = 0.40)
        # plot input graph
        nx.draw_networkx(graphLeaf, with_labels = True, labels = mapToRoot[eachLeaf],
                         pos = posLeaf, node_size = 150, nodelist = nodeList, node_color = nodeColor,
                         font_size = 7, edge_color = "k", width = 2)
        plt.ylabel("Input Graph " + eachLeaf, fontsize = 14, weight = "light")
        plt.xlim([minX, maxX])
        plt.ylim([minY, maxY])
        plt.tight_layout()
        # save page
        sanityCheckPDF.savefig(fig)
        # clear current figure
        plt.clf()
        # save alignmet 3D information
        if(vis3D):
            tupleInfo3D = []
            tupleInfo3D.append(graphLeaf.order())
            tupleInfo3D.append(eachLeaf)
            tupleInfo3D.append(deepcopy(posLeaf))
            tupleInfo3D.append(nodeList)
            tupleInfo3D.append(nodeColor)
            tupleInfo3D.append(deepcopy(graphLeaf))
            tupleInfo3D.append(deepcopy(mapToRoot[eachLeaf]))
            graphFigs3DInfo.append(deepcopy(tuple(tupleInfo3D)))
    # create figure
    fig = plt.figure(figsize = [wOffset + vExtra, hOffset + vExtra])
    # plot alignment
    nx.draw_networkx(finalAlignment, with_labels = True,
                     pos = posAlignment, node_size = 150, node_color = "silver",
                     font_size = 7, edge_color = "k", width = 2)
    plt.ylabel("Alignment", fontsize = 14, weight = "light")
    plt.xlim([minX, maxX])
    plt.ylim([minY, maxY])
    plt.tight_layout()
    # save page
    sanityCheckPDF.savefig(fig)
    # save pdf
    sanityCheckPDF.close()
    plt.close()


# 3D visualization
if(vis3D):
    # save data for 3D plots
    outputFile = open(name3DPlotsPKL, "wb")
    pkl.dump(graphFigs3DInfo, outputFile)
    outputFile.close()


# 1D visualization
if(vis1D):
    # create file
    sanityCheckPDF = PdfPages(name1DPlotsPDF)
    # clear current figure
    plt.clf()
    # create figure
    fig = plt.figure(figsize = [wOffset + vExtra, hOffset + vExtra])
    # draw final alignment
    nx.draw_networkx(finalAlignment, with_labels = True,
                     pos = posAlignment, node_size = 150,
                     font_size = 6, node_color = "silver",
                     edge_color = "k", width = 2)
    plt.ylabel("Alignment", fontsize = 14, weight = "light")
    plt.xlim([minX, maxX])
    plt.ylim([minY, maxY])
    plt.tight_layout()
    # save page
    sanityCheckPDF.savefig(fig)
    # clear current figure
    plt.clf()
    # create figure [width, height] in inches
    fig = plt.figure(figsize = [wOffset + vExtra, hOffset + gExtra])
    # build matrix
    nodePresence = dict()
    backMap = dict()
    for eachLeaf in allLeaves:
        graphLeaf = deepcopy(allLeavesGraphs[eachLeaf])
        backMap[eachLeaf] = {mapToRoot[eachLeaf][u]:u for u in list(graphLeaf.nodes())}
    for v in nodesAlignment:
        nodePresence[v] = []
        for eachLeaf in allLeaves:
            if(not existence[dendrogramSSP][v][eachLeaf] == "-"):
                graphLeaf = deepcopy(allLeavesGraphs[eachLeaf])
                tempDict = deepcopy(graphLeaf.nodes[backMap[eachLeaf][v]]["nodeLabel"])
                nodeInfoTuple = [(labelName, tempDict[labelName]) for labelName in list(tempDict.keys())]
                nodeInfoTuple.sort()
                nodeInfoTuple = tuple(nodeInfoTuple)
                nodePresence[v].append(labelColor[nodeInfoTuple])
            else:
                nodePresence[v].append(np.nan)
    # draw heatmap of alignment
    finalArray = [nodePresence[v] for v in nodesAlignment]
    maskedArray = np.ma.array(finalArray, mask = np.isnan(finalArray))
    maskedArray = np.transpose(maskedArray)
    im = plt.imshow(maskedArray, cmap = myCmap, norm = myNorm, aspect = "equal")
    # set minor ticks
    ax = plt.gca()
    # set major tick positions
    ax.set_xticks(np.arange(len(nodesAlignment)))
    ax.set_yticks(np.arange(len(allLeaves)))
    # set major tick labels
    ax.set_xticklabels(nodesAlignment, fontsize = 5.5)
    ax.set_yticklabels(allLeaves, fontsize = 5.5)
    # set minor ticks
    ax.set_xticks(np.arange(-0.5, len(nodesAlignment), 1), minor = True)
    ax.set_yticks(np.arange(-0.5, len(allLeaves), 1), minor = True)
    # loop over data dimensions and create text annotations
    # for i in range(len(allLeaves)):
    #     for j in range(len(nodesAlignment)):
    #         try:
    #             ax.text(j, i, str(int(maskedArray[i, j])), ha = "left", va = "bottom", color = "k", fontsize = 4)
    #         except:
    #             pass
    # set grid
    ax.grid(which = "minor", color = "k", linestyle = "-", linewidth = 1)
    # remove minor ticks
    ax.tick_params(which = "minor", bottom = False, left = False)
    # finish image
    plt.title("Matrix Representation of the Alignment\n", fontsize = 12, weight = "light")
    plt.xlabel("\nVertices of the alignment", fontsize = 10, weight = "light")
    plt.ylabel("Input Graphs\n", fontsize = 10, weight = "light")
    # save page
    sanityCheckPDF.savefig(fig)
    # clear current figure
    plt.clf()
    # create figure
    fig = plt.figure(figsize = [wOffset, hOffset])
    # plot color code information as legend
    legendInfo = dict()
    for labelTuple in list(labelColor.keys()):
        strLabels = []
        for (labelName, labelValue) in labelTuple:
            strLabels.append("(" + str(labelName) + ", " + str(labelValue) + ")")
        strLabelFinal = ", ".join(strLabels)
        legendInfo[strLabelFinal] = labelColor[labelTuple]
    for eachStr in list(legendInfo.keys()):
        if(eachStr == ""):
            plt.plot([0], marker = "s", linestyle = ":", color = colorCode[legendInfo[eachStr]], label = str(legendInfo[eachStr]) + ":  $\it{unlabeled}$")
        else:
            plt.plot([0], marker = "s", linestyle = ":", color = colorCode[legendInfo[eachStr]], label = str(legendInfo[eachStr]) + ": " + eachStr)
    plt.plot([0], marker = "s", linestyle = ":", color = "w", markersize = 10)
    # make legend box
    plt.legend(loc = "upper left", fontsize = 6, framealpha = 1)
    # set major tick positions
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    # drawing properties
    plt.title("Color code of vertex labels", fontsize = 8, weight = "light")
    plt.tight_layout()
    # save page
    sanityCheckPDF.savefig(fig)
    # save pdf
    sanityCheckPDF.close()
    plt.close()


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
