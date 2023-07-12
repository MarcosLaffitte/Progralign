################################################################################
#                                                                              #
#  README - Program: Progralign_Vis3D.py                                       #
#                                                                              #
#  - Paper: https://www.mdpi.com/1999-4893/17/3/116                            #
#                                                                              #
#  - Github repository: https://github.com/MarcosLaffitte/Progralign           #
#                                                                              #
#  - Date: 12 March 2024                                                       #
#                                                                              #
#  - Contributor(s) to this script:                                            #
#  * @MarcosLaffitte - Marcos E. Gonzalez Laffitte                             #
#                                                                              #
#  - Description: receives a file with data to make a 3D plots and 3D          #
#    animations of an alignment produced by Progralign_Vis2D in the repo.      #
#                                                                              #
#  - Input: example_Results_3D_data.pkl; the angles and parameters of the 3D   #
#    plots and animation can be changed in-code.                               #
#                                                                              #
#  - Run with (after activating [pgalign] conda environment):                  #
#  * default  python Progralign_Vis3D.py example_Results_3D_data.pkl           #
#  * or       python Progralign_Vis3D.py --anim example_Results_3D_data.pkl    #
#                                                                              #
#  - Expected output:                                                          #
#  * (1) example_Results_3D_plots.png                                          #
#  * (2) example_Results_3D_plots_ordered.png                                  #
#  * (3) example_Results_3D_movie.gif            (only with --anim option)     #
#  * (4) example_Results_3D_movie_ordered.gif    (only with --anim option)     #
#                                                                              #
#  NOTE: the 3D plots made by Progralign_Vis3D are made by stacking the 2D     #
#  plots that are the output of Progralign_Vis2D, and this cannot be changed   #
#  here, only directly in Progralign_Vis2D.                                    #
#                                                                              #
#  WARNING! make sure there are no png files in the diretcory when running     #
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
from mpl_toolkits import mplot3d
import matplotlib.animation as animation


# already in python ------------------------------------------------------------
import os
import pickle as pkl
from copy import deepcopy
from sys import argv, exit
from math import modf, sqrt


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# input and variables ##########################################################


# 3D plot user defined-parameters ----------------------------------------------
strideVal = 1
levelSkip = 3    # number of "faux" figures between plots
elevAngle = 18
azimAngle = -120
maxGraphs = 6


# list of angles for frames ----------------------------------------------------
angleVals = [(eachElev, 90) for eachElev in range(1, 15, 2)]
angleVals = angleVals + [(15, 90-eachMinus) for eachMinus in range(0, 50, 5)]
angleVals = angleVals + [(eachElev, 45) for eachElev in range(17, 27, 2)]
angleVals = angleVals + [(25, eachAzim) for eachAzim in range(50, 90, 5)] + [(25, 90)]
angleVals = [(angleVals[index][0], angleVals[index][1], index+1, len(angleVals)) for index in range(len(angleVals))]


# check input ------------------------------------------------------------------
if(len(argv) in [2, 3]):
    if(len(argv) == 2):
        if(".pkl" in argv[1]):
            remainder = (argv[1].split(".pkl"))[-1]
            if(not remainder == ""):
                errorStr = "\n >> Progralign_Vis3D: wrong input extension.\n"
                errorStr = errorStr + "- Expected: *.pkl\n"
                errorStr = errorStr + "- Received: *.pkl" + remainder + "\n"
                exit(errorStr)
            else:
                inputFileName = argv[1]
                createAnimation = False
        else:
            exit("\n >> Progralign: wrong input format.\n")
    if(len(argv) == 3):
        if((argv[1] == "--anim") and (".pkl" in argv[2])):
            remainder = (argv[2].split(".pkl"))[-1]
            if(not remainder == ""):
                errorStr = "\n >> Progralign: wrong input extension.\n"
                errorStr = errorStr + "- Expected: *.pkl\n"
                errorStr = errorStr + "- Received: *.pkl" + remainder + "\n"
                exit(errorStr)
            else:
                inputFileName = argv[2]
                createAnimation = True
        else:
            exit("\n >> Progralign: wrong input format.\n")
else:
    exit("\n >> Progralign: wrong input format.\n")


# output -----------------------------------------------------------------------
name3DPlotsPNG = inputFileName.replace("_3D_data.pkl", "_3D_plots.png")
name3DPlotsOrderedPNG = inputFileName.replace("_3D_data.pkl", "_3D_plots_ordered.png")
name3DMoviePNG = inputFileName.replace("_3D_data.pkl", "_3D_movie.gif")
name3DMovieOrderedPNG = inputFileName.replace("_3D_data.pkl", "_3D_movie_ordered.gif")


# functions - GENERAL TASKS ####################################################


# function: print custom progress bar ------------------------------------------
def printProgress(casePercentage, caseNum = 0, totCases = 0, reportCase = True, uTurn = True):
    # local variables
    tail = "".join(10*[" "])
    base = "-"
    done = "="
    bar = ""
    pile = []
    finished = ""
    percentageInt = 0
    # generate bar
    percentageInt = int(modf(casePercentage/10)[1])
    for i in range(1, 11):
        if(i <= percentageInt):
            pile.append(done)
        else:
            pile.append(base)
    finished = "".join(pile)
    if(reportCase):
        bar = "- progress:   0%  [" + finished + "]  100%" + " ;  done frame: " + str(caseNum) + " / " + str(totCases)
    else:
        bar = "- progress:   0%  [" + finished + "]  100%"
    # message
    if(uTurn):
        print(bar + tail, end = "\r")
    else:
        print(bar + tail)


# function: move angles in animation -------------------------------------------
def moveAngles(frame):
    # local variables
    elevVal = frame[0]
    azimVal = frame[1]
    ax.view_init(elev = elevVal, azim = azimVal)
    # print progress
    frameNum = frame[2]
    frameTot = frame[3]
    printProgress(round(frameNum*100/frameTot, 2), caseNum = frameNum, totCases = frameTot)
    # end of function


# main #########################################################################


# initial message
print("\n")
print(">>> Progralign 3D Visualization - Progralign Github Repository")


# task message
print("\n")
print("* Retrieving input file ...")


# open pickle again
inputFile = open(inputFileName, "rb")
inputList = pkl.load(inputFile)
inputFile.close()


# check length of input data
if(len(inputList) > maxGraphs):
    errorStr = "\n*** Wrong Input: Received " + str(len(inputList)) + " graphs.\n"
    errorStr = errorStr + "*** The 3D plots can be done with at most " + str(maxGraphs) + " graphs.\n"
    exit(errorStr)


# task message
print("\n")
print("* Making elementary 2D plots ...")


# make 2D plots while following input order
figureNumber = 0
for (eachOrder, eachName, eachPos, eachNodes, eachColor, eachGraph, eachLabels) in inputList:
    # set parameters
    minX = -1.15
    maxX = 1.15
    minY = -1.15
    maxY = 1.15
    # create figure
    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw_networkx(eachGraph, with_labels = True,
                     labels = eachLabels, pos = eachPos, node_size = 200,
                     nodelist = eachNodes, node_color = eachColor,
                     font_size = 8, edge_color = "k", width = 0.5)
    plt.xlim([minX, maxX])
    plt.ylim([minY, maxY])
    plt.tight_layout()
    ax.axis("off")
    # save figure
    plt.savefig("Fig_" + str(figureNumber) + ".png", transparent = True)
    plt.close()
    figureNumber = figureNumber + 1


# create false figure
fig = plt.figure()
ax = fig.add_subplot()
fauxG = nx.Graph()
nx.draw_networkx(fauxG)
plt.xlim([minX, maxX])
plt.ylim([minY, maxY])
plt.tight_layout()
ax.axis("off")
plt.savefig("faux.png", dpi = 1, transparent = True)
plt.close()


# create plotting array
plotArray = []
for i in range(1, len(inputList)):
    for k in range(levelSkip):
        plotArray.append("./faux.png")
    plotArray.append("./Fig_" + str(i) + ".png")
plotArray.append("./faux.png")


# task message
print("\n")
print("* Creating 3D plot following input order ...")


# add alignment to 3D plot
levelZ = 0
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.set_zlim3d(bottom = levelZ-1, top = len(plotArray)+1)
# open 2D plot of each graph
imgArray = plt.imread("./Fig_0.png")
imgArray = np.swapaxes(imgArray, 0, 1)
imgArray = np.flip(imgArray, 1)
# get X and Y data arrrays
stepX = 10 / imgArray.shape[0]
stepY = 10 / imgArray.shape[1]
valsX = np.arange(0, 10, stepX)
valsY = np.arange(0, 10, stepY)
valsX, valsY = np.meshgrid(valsX, valsY)
# get Z data arrrays
valsZ = np.ones((imgArray.shape[1], imgArray.shape[0]))*levelZ
# last flip to image necessary for some reason
imgArray = np.swapaxes(imgArray, 0, 1)
# add graph to 3D plot
ax.plot_surface(valsX, valsY, valsZ,
                rstride = strideVal, cstride = strideVal,
                facecolors = imgArray)


# add input graphs to 3D plot
for imgName in plotArray:
    # open 2D plot of each graph
    imgArray = plt.imread(imgName)
    imgArray = np.swapaxes(imgArray, 0, 1)
    imgArray = np.flip(imgArray, 1)
    # get X and Y data arrrays
    stepX = 10 / imgArray.shape[0]
    stepY = 10 / imgArray.shape[1]
    valsX = np.arange(0, 10, stepX)
    valsY = np.arange(0, 10, stepY)
    valsX, valsY = np.meshgrid(valsX, valsY)
    # get Z data arrrays
    levelZ = levelZ + 1
    valsZ = np.ones((imgArray.shape[1], imgArray.shape[0]))*levelZ
    # last flip to image necessary for some reason
    imgArray = np.swapaxes(imgArray, 0, 1)
    # add graph to 3D plot
    if(imgName == "./faux.png"):
        ax.plot_surface(valsX, valsY, valsZ,
                        rstride = 50, cstride = 50,
                        facecolors = imgArray)
    else:
        ax.plot_surface(valsX, valsY, valsZ,
                        rstride = strideVal, cstride = strideVal,
                        facecolors = imgArray)
# figure attributes
ax.set(title = "Graph Alignment")
ax.set(xticklabels = [],
       yticklabels = [],
       zticklabels = [])
ax.set(xticks = [],
       yticks = [],
       zticks = [])
ax.grid(False)
ax.view_init(elev = elevAngle, azim = azimAngle)
plt.tight_layout()
# save figure with basic angles
plt.savefig(name3DPlotsPNG, bbox_inches = "tight", dpi = 600)


# task message
if(createAnimation):
    print("\n")
    print("* Creating animation following input order ...")


# make animation
if(createAnimation):
    ax.set(title = "")
    movie = animation.FuncAnimation(fig, moveAngles, frames = angleVals, interval = 300)
    movie.save(name3DMoviePNG, writer = "imagemagick", dpi = 200)
plt.close()


# task message
print("\n")
print("* Done ...")


# erease 2D plots
plotArray = ["./Fig_0.png"] + list(set(plotArray))
for imgName in plotArray:
    os.system("rm " + imgName)


# reverse order of input graphs
alignment = deepcopy(inputList[0])
inputList = inputList[1:]
inputList.sort(reverse = True)
inputList = [alignment] + inputList


# task message
print("\n")
print("* Creating 3D plot now following decreasing number of vertices ...")


# make 2D plots following drécreasing order
figureNumber = 0
for (eachOrder, eachName, eachPos, eachNodes, eachColor, eachGraph, eachLabels) in inputList:
    # set parameters
    minX = -1.15
    maxX = 1.15
    minY = -1.15
    maxY = 1.15
    # create figure
    fig = plt.figure()
    ax = fig.add_subplot()
    nx.draw_networkx(eachGraph, with_labels = True,
                     labels = eachLabels, pos = eachPos, node_size = 200,
                     nodelist = eachNodes, node_color = eachColor,
                     font_size = 8, edge_color = "k", width = 0.5)
    plt.xlim([minX, maxX])
    plt.ylim([minY, maxY])
    plt.tight_layout()
    ax.axis("off")
    # save figure
    plt.savefig("Fig_" + str(figureNumber) + ".png", transparent = True)
    plt.close()
    figureNumber = figureNumber + 1


# create false figure
fig = plt.figure()
ax = fig.add_subplot()
fauxG = nx.Graph()
nx.draw_networkx(fauxG)
plt.xlim([minX, maxX])
plt.ylim([minY, maxY])
plt.tight_layout()
ax.axis("off")
plt.savefig("faux.png", dpi = 1, transparent = True)
plt.close()


# create plotting array
plotArray = []
for i in range(1, len(inputList)):
    for k in range(levelSkip):
        plotArray.append("./faux.png")
    plotArray.append("./Fig_" + str(i) + ".png")
plotArray.append("./faux.png")


# add alignment to 3D plot
levelZ = 0
fig = plt.figure()
ax = fig.add_subplot(projection = "3d")
ax.set_zlim3d(bottom = levelZ-1, top = len(plotArray)+1)
# open 2D plot of each graph
imgArray = plt.imread("./Fig_0.png")
imgArray = np.swapaxes(imgArray, 0, 1)
imgArray = np.flip(imgArray, 1)
# get X and Y data arrrays
stepX = 10 / imgArray.shape[0]
stepY = 10 / imgArray.shape[1]
valsX = np.arange(0, 10, stepX)
valsY = np.arange(0, 10, stepY)
valsX, valsY = np.meshgrid(valsX, valsY)
# get Z data arrrays
valsZ = np.ones((imgArray.shape[1], imgArray.shape[0]))*levelZ
# last flip to image necessary for some reason
imgArray = np.swapaxes(imgArray, 0, 1)
# add graph to 3D plot
ax.plot_surface(valsX, valsY, valsZ,
                rstride = strideVal, cstride = strideVal,
                facecolors = imgArray)


# add input graphs to 3D plot
for imgName in plotArray:
    # open 2D plot of each graph
    imgArray = plt.imread(imgName)
    imgArray = np.swapaxes(imgArray, 0, 1)
    imgArray = np.flip(imgArray, 1)
    # get X and Y data arrrays
    stepX = 10 / imgArray.shape[0]
    stepY = 10 / imgArray.shape[1]
    valsX = np.arange(0, 10, stepX)
    valsY = np.arange(0, 10, stepY)
    valsX, valsY = np.meshgrid(valsX, valsY)
    # get Z data arrrays
    levelZ = levelZ + 1
    valsZ = np.ones((imgArray.shape[1], imgArray.shape[0]))*levelZ
    # last flip to image necessary for some reason
    imgArray = np.swapaxes(imgArray, 0, 1)
    # add graph to 3D plot
    if(imgName == "./faux.png"):
        ax.plot_surface(valsX, valsY, valsZ,
                        rstride = 50, cstride = 50,
                        facecolors = imgArray)
    else:
        ax.plot_surface(valsX, valsY, valsZ,
                        rstride = strideVal, cstride = strideVal,
                        facecolors = imgArray)
# figure attributes
ax.set(title = "Graph Alignment")
ax.set(xticklabels = [],
       yticklabels = [],
       zticklabels = [])
ax.set(xticks = [],
       yticks = [],
       zticks = [])
ax.grid(False)
ax.view_init(elev = elevAngle, azim = azimAngle)
plt.tight_layout()
# save figure with basic angles
plt.savefig(name3DPlotsOrderedPNG, bbox_inches = "tight", dpi = 600)


# task message
if(createAnimation):
    print("\n")
    print("* Creating animation now following decreasing number of vertices ...")


# make animation
if(createAnimation):
    ax.set(title = "")
    movie = animation.FuncAnimation(fig, moveAngles, frames = angleVals, interval = 300)
    movie.save(name3DMovieOrderedPNG, writer = "imagemagick", dpi = 200)
plt.close()


# erease 2D plots
plotArray = ["./Fig_0.png"] + list(set(plotArray))
for imgName in plotArray:
    os.system("rm " + imgName)


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
