################################################################################
#                                                                              #
#  README - Program: Progralign_Analysis.py                                    #
#                                                                              #
#  - Description: receives a list of arbitrary (possibly labeled) NetworkX     #
#    graphs (all of the same type, be it undirected or directed) saved in a    #
#    "*.pkl" file, and carries the progressive alignment of these graphs by    #
#    following a guide tree built with WPGMA on graph kernel similarities.     #
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
import gklearn.kernels.structuralspKernel as sspk
from matplotlib.backends.backend_pdf import PdfPages


# already in python ------------------------------------------------------------
import time
import random
import pickle as pkl
from copy import deepcopy
from math import modf, sqrt
from operator import eq, itemgetter
from itertools import product, combinations
from sys import argv, exit, getrecursionlimit, setrecursionlimit


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# input and variables ##########################################################


# use input for drawing, funtionalities and heuristics -------------------------
upperBound3D = 5
upperBound2D = 20
upperBound1D = 40
# (default) "match": maximize sum of scores; or "order": maximize number of vertices
requestedScore = "match"
# (default) "distance": guide tree built on distance; or "similarity": built on the value returned by the kernel
guideTreeMeasure = "distance"
# (default) b.True: carry analysis considering ambiguous edges of every inner node of the guide-tree
ambiguousEdges = True
# (default) b.True: "room-for-improvement (ROFI) pruning" heuristic for reducing MCS running time
heuristicROFI = False
# (default) b.True: only allow alignment of vertices with same labels
respectVLabels = True
# (default) b.True: only allow alignment of edges with same labels
respectELabels = True
# (default) b.True: self pairwise-alignment score-scheme is equivalent to matched vertices with same label
basicSelfScoring = True
# (default) b.True: finish Match search if one (sub)graph isomorphism is found as match and return such match
justOneSubIso = True
# (default) b.True: if justOneSubIso is True and respectVLabels False then the required subIso can be unlabeld
subIsoMustBeLabeled = True
# (default) b.True: finish Match search if one (maximal) match inducing unmatched vertices with disjoint label-sets is found
justOneMaximalColoredMatch = True


# input ------------------------------------------------------------------------
inputFileName = ""
inputList = []
inputFile = None
drawingMode = False


# data holders -----------------------------------------------------------------
counter = 0
heightGT = 0
leafCount = 0
finalScore = 0
scoreDistPM = 0
scoreDistMM = 0
selfScoreG1 = 0
selfScoreG2 = 0
kernelDistPM = 0
kernelDistMM = 0
pairScoreG1G2 = 0
uvDistanceSSP = 0
uvSimilaritySSP = 0
theType = ""
newName = ""
dendrogramSSP = ""
match = []
indices = []
cherries = []
toRemove = []
leafNeighbors= []
notMatchNodes = []
matchingNodes = []
pairsOfMutants = []
scoreDistancePMVals = []
scoreDistanceMMVals = []
kernelDistancePMVals = []
kernelDistanceMMVals = []
allScores = dict()
existence = dict()
mapToRoot = dict()
alignments = dict()
mapToParent = dict()
depthLeaves = dict()
namesByIndex = dict()
ambiguousPairs = dict()
scoreDistancePM = dict()
scoreDistanceMM = dict()
kernelDistancePM = dict()
kernelDistanceMM = dict()
existencePrimalComp = dict()
existenceMutantComp = dict()
isMatchAlsoSubIso = False
isMatchAlsoMaxColored = False
G1 = None
G2 = None
primal = None
guideTree = None
condPrimal = None
guideTreeSSP = None
finalAlignment = None
kernelMatrixSSP  = None
clusteringGraphSSP = None


# time data --------------------------------------------------------------------
finalTime = 0
initialTime = 0
alignmentTime = 0


# check input ------------------------------------------------------------------
if(len(argv) in [2, 3]):
    if(len(argv) == 2):
        if(".pkl" in argv[1]):
            remainder = (argv[1].split(".pkl"))[-1]
            if(not remainder == ""):
                errorStr = "\n >> Progralign: wrong input extension.\n"
                errorStr = errorStr + "- Expected: *.pkl\n"
                errorStr = errorStr + "- Received: *.pkl" + remainder + "\n"
                exit(errorStr)
            else:
                inputFileName = argv[1]
                drawingMode = False
        else:
            exit("\n >> Progralign: wrong input format.\n")
    if(len(argv) == 3):
        if((argv[1] == "--draw") and (".pkl" in argv[2])):
            remainder = (argv[2].split(".pkl"))[-1]
            if(not remainder == ""):
                errorStr = "\n >> Progralign: wrong input extension.\n"
                errorStr = errorStr + "- Expected: *.pkl\n"
                errorStr = errorStr + "- Received: *.pkl" + remainder + "\n"
                exit(errorStr)
            else:
                inputFileName = argv[2]
                drawingMode = True
        else:
            exit("\n >> Progralign: wrong input format.\n")
else:
    exit("\n >> Progralign: wrong input format.\n")


# output -----------------------------------------------------------------------
outputFile = None
constructionDict = dict()
name2DPlotsPDF = inputFileName.replace(".pkl", "_2D_plot.pdf")
name1DPlotsPDF = inputFileName.replace(".pkl", "_1D_plot.pdf")
name3DPlotsPKL = inputFileName.replace(".pkl", "_3D_data.pkl")
nameResultsPKL = inputFileName.replace(".pkl", "_Results.pkl")
nameScatterPM = inputFileName.replace(".pkl", "_Scatter_PM.pdf")
nameScatterMM = inputFileName.replace(".pkl", "_Scatter_MM.pdf")


# functions - GENERAL TASKS ####################################################


# function: print custom progress bar ------------------------------------------
def printProgress(casePercentage, caseNum = 0, totCases = 0, progressIn = "", reportCase = True, uTurn = True):
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
        if(progressIn == ""):
            bar = "- progress:  0%  [" + finished + "]  100%" + " ;  done: " + str(caseNum) + " / " + str(totCases)
        else:
            bar = "- progress " + progressIn + ":  0%  [" + finished + "]  100%" + " ;  done: " + str(caseNum) + " / " + str(totCases)
    else:
        if(progressIn == ""):
            bar = "- progress:  0%  [" + finished + "]  100%"
        else:
            bar = "- progress " + progressIn + ":  0%  [" + finished + "]  100%"
    # message
    if(uTurn):
        print(bar + tail, end = "\r")
    else:
        print(bar + tail)


# function: condense labels into one dict for comparison -----------------------
def condensedLabel(someG):
    # local variables
    uniformG = None
    # get graph of corresponding type
    if(someG.is_directed()):
        uniformG = nx.DiGraph()
    else:
        uniformG = nx.Graph()
    # iterate over nodes condensing their labels
    for (v, nodeInfo) in list(someG.nodes(data = True)):
        uniformG.add_node(v, nodeLabel = nodeInfo)
    # iterate over edges condensing their labels
    for (u, v, edgeInfo) in list(someG.edges(data = True)):
        uniformG.add_edge(u, v, edgeLabel = edgeInfo)
    # end of function
    return(uniformG)


# function: extend condensed labels inside dict into single labels -------------
def extendedLabel(someG):
    # local variables
    extendedG = None
    nodeAttributes = dict()
    edgeAttributes = dict()
    # get graph of corresponding type
    if(someG.is_directed()):
        extendedG = nx.DiGraph()
    else:
        extendedG = nx.Graph()
    # get attributes of nodes
    for (v, nodeInfo) in list(someG.nodes(data = True)):
        extendedG.add_node(v)
        nodeAttributes[v] = deepcopy(nodeInfo["nodeLabel"])
    # get attributes of edges
    for (u, v, edgeInfo) in list(someG.edges(data = True)):
        extendedG.add_edge(u, v)
        edgeAttributes[(u, v)] = deepcopy(edgeInfo["edgeLabel"])
    # asign new attributes
    nx.set_node_attributes(extendedG, nodeAttributes)
    nx.set_edge_attributes(extendedG, edgeAttributes)
    # end of function
    return(extendedG)


# function: condense labels into one single string -----------------------------
def condensedLabelToStr(someG):
    # local variables
    strLabels = []
    strLabelFinal = ""
    uniformG = None
    nodeInfoTuple = None
    edgeInfoTuple = None
    # get graph of corresponding type
    if(someG.is_directed()):
        uniformG = nx.DiGraph()
    else:
        uniformG = nx.Graph()
    # iterate over nodes condensing their labels into tuple
    for (v, nodeInfo) in list(someG.nodes(data = True)):
        nodeInfoTuple = [(labelName, nodeInfo[labelName]) for labelName in list(nodeInfo.keys())]
        nodeInfoTuple.sort()
        nodeInfoTuple = tuple(nodeInfoTuple)
        if(len(nodeInfoTuple) > 0):
            strLabels = []
            for (labelName, labelValue) in nodeInfoTuple:
                strLabels.append("(" + str(labelName) + ", " + str(labelValue) + ")")
            strLabelFinal = ", ".join(strLabels)
        else:
            strLabelFinal = "empty"
        uniformG.add_node(v, nodeLabelStr = strLabelFinal)
    # iterate over edges condensing their labels into tuple
    for (u, v, edgeInfo) in list(someG.edges(data = True)):
        edgeInfoTuple = [(labelName, edgeInfo[labelName]) for labelName in list(edgeInfo.keys())]
        edgeInfoTuple.sort()
        edgeInfoTuple = tuple(edgeInfoTuple)
        if(len(edgeInfoTuple) > 0):
            strLabels = []
            for (labelName, labelValue) in edgeInfoTuple:
                strLabels.append("(" + str(labelName) + ", " + str(labelValue) + ")")
            strLabelFinal = ", ".join(strLabels)
        else:
            strLabelFinal = "empty"
        uniformG.add_edge(u, v, edgeLabelStr = strLabelFinal)
    # end of function
    return(uniformG)


# function: compute dendrogram based on similarity or distance -----------------
def getGuideTree(someClusteringGraphSSP, someTreeMeasure = "distance"):
    # local variables
    optimalMeasure = 0
    newMeasure = 0
    dendroNode = ""
    theDendrogram = ""
    candidates = []
    otherNodes = []
    measures = dict()
    theContainedGraphs = dict()
    theGuideTree = None
    dendroGraph = None
    # create guide tree and add leaves
    theGuideTree = nx.Graph()
    theGuideTree.add_nodes_from(list(someClusteringGraphSSP.nodes()))
    # save leaves as contained graphs
    for eachLeaf in list(someClusteringGraphSSP.nodes()):
        theContainedGraphs[eachLeaf] = [eachLeaf]
    # iterative construction of dendrogram
    dendroGraph = deepcopy(someClusteringGraphSSP)
    while(dendroGraph.order() > 1):
        # get measures and optimal measure
        measures = nx.get_edge_attributes(dendroGraph, "measure")
        if(someTreeMeasure == "distance"):
            optimalMeasure = min(list(measures.values()))
        if(someTreeMeasure == "similarity"):
            optimalMeasure = max(list(measures.values()))
        # get candidate edges to contract and get edge to be compressed
        candidates = [(u, v) for (u, v) in list(dendroGraph.edges()) if(dendroGraph[u][v]["measure"] == optimalMeasure)]
        (x, y) = tuple(sorted(list(candidates[0])))
        # update dendroGraph contracting vertices into dendroNode
        dendroNode = "(" + x + "," + y + ")"
        dendroGraph.add_node(dendroNode)
        otherNodes = [v for v in list(dendroGraph.nodes()) if(not v in [x, y, dendroNode])]
        for v in otherNodes:
            newMeasure = (dendroGraph[x][v]["measure"] + dendroGraph[y][v]["measure"])/2
            dendroGraph.add_edge(dendroNode, v, measure = newMeasure)
        dendroGraph.remove_node(x)
        dendroGraph.remove_node(y)
        # add dendroNode to guide tree
        theGuideTree.add_node(dendroNode)
        theGuideTree.add_edge(x, dendroNode)
        theGuideTree.add_edge(y, dendroNode)
        # saved leaves hanging in subtree
        theContainedGraphs[dendroNode] = theContainedGraphs[x] + theContainedGraphs[y]
    # end of function
    theDendrogram = list(dendroGraph.nodes())[0]
    return(theDendrogram, theGuideTree, theContainedGraphs)


# function: build alignment (di-)graph -----------------------------------------
def buildAlignment(someG1, someG2, someMatch, existenceG1, existenceG2, containedG1, containedG2):
    # local variables
    k = 0
    classA = []   # class A: MCS in G1
    classB = []   # class B: complement of MCS in G1
    classC = []   # class C: complement of MCS in G2
    classD = []   # class D: MCS in G2
    nodesG1 = []
    nodesG2 = []
    edgesG1 = []
    edgesG2 = []
    repeatedEdges = []
    copyG1 = None
    copyG2 = None
    theAlignment = None
    invM = dict()
    forwardMatch = dict()
    theExistence = dict()
    theMapToParentG1 = dict()
    theMapToParentG2 = dict()
    # get vertices and classes
    nodesG1 = list(someG1.nodes())
    nodesG2 = list(someG2.nodes())
    classA = [v1 for (v1, v2) in someMatch]
    classD = [v2 for (v1, v2) in someMatch]
    classB = [v for v in nodesG1 if(not v in classA)]
    classC = [v for v in nodesG2 if(not v in classD)]
    # sort vertices and classes
    nodesG1.sort()
    nodesG2.sort()
    classA.sort()
    classB.sort()
    classC.sort()
    classD.sort()
    # new labels for vertices
    k = 0
    for v in classA:
        theMapToParentG1[v] = k
        k = k + 1
    for v in classB:
        theMapToParentG1[v] = k
        k = k + 1
    for v in classC:
        theMapToParentG2[v] = k
        k = k + 1
    for v in classD:
        theMapToParentG2[v] = k
        k = k + 1
    # create copies and remove "repeated" edges in G2 (that are not matching ambiguous edges)
    edgesG1 = list(someG1.edges())
    edgesG2 = list(someG2.edges())
    invM = {n2:n1 for (n1, n2) in someMatch}
    copyG1 = deepcopy(someG1)
    copyG1 = nx.relabel_nodes(copyG1, theMapToParentG1)
    copyG2 = deepcopy(someG2)
    repeatedEdges = [(u, v) for (u, v) in edgesG2 if((u in classD) and (v in classD))]
    repeatedEdges = [(u, v) for (u, v) in repeatedEdges if(((invM[u], invM[v]) in edgesG1) or ((invM[v], invM[u]) in edgesG1))]
    copyG2.remove_edges_from(repeatedEdges)
    copyG2 = nx.relabel_nodes(copyG2, theMapToParentG2)
    # make union
    theAlignment = nx.union(copyG1, copyG2)
    # contract vertices in match
    for (v1, v2) in someMatch:
        theAlignment = nx.identified_nodes(theAlignment, theMapToParentG1[v1], theMapToParentG2[v2])
        theMapToParentG2[v2] = theMapToParentG1[v1]
    # obtain existence
    theExistence = dict()
    theLeaves = list(list(existenceG1.values())[0].keys())
    forwardMatch = {v1:v2 for (v1, v2) in someMatch}
    for v in classA:
        vP = theMapToParentG1[v]
        theExistence[vP] = dict()
        for leaf in theLeaves:
            theExistence[vP][leaf] = "."
            if(leaf in containedG1):
                theExistence[vP][leaf] = deepcopy(existenceG1[v][leaf])
            if(leaf in containedG2):
                theExistence[vP][leaf] = deepcopy(existenceG2[forwardMatch[v]][leaf])
    for v in classB:
        vP = theMapToParentG1[v]
        theExistence[vP] = dict()
        for leaf in theLeaves:
            theExistence[vP][leaf] = "."
            if(leaf in containedG1):
                theExistence[vP][leaf] = deepcopy(existenceG1[v][leaf])
            if(leaf in containedG2):
                theExistence[vP][leaf] = "-"
    for v in classC:
        vP = theMapToParentG2[v]
        theExistence[vP] = dict()
        for leaf in theLeaves:
            theExistence[vP][leaf] = "."
            if(leaf in containedG1):
                theExistence[vP][leaf] = "-"
            if(leaf in containedG2):
                theExistence[vP][leaf] = deepcopy(existenceG2[v][leaf])
    # end of function
    return((theAlignment, theExistence, theMapToParentG1, theMapToParentG2))


# function: scoring function for MCS optimization ------------------------------
def matchScore(someG1, someG2, someMatch, existenceG1, existenceG2, containedG1, containedG2, score = "match"):
    # local variables
    theScore = 0
    # get requested score
    if(score == "order"):
        # optimize MCS based on number of vertices
        theScore = len(someMatch)
    if(score == "match"):
        # optimize MCS based on sum of scores
        k = 0
        classA = []
        classB = []
        classC = []
        classD = []
        nodesG1 = []
        nodesG2 = []
        forwardMatch = dict()
        theMapToParentG1 = dict()
        theMapToParentG2 = dict()
        # score scheme for arbitrary labels (no prefered order)
        scoreSameLabel = 1
        scoreDiffLabel = 0
        scoreLabelGap = 0
        scoreGapGap = 0
        # get relevant sets of vertices
        nodesG1 = list(someG1.nodes())
        nodesG2 = list(someG2.nodes())
        classA = [v1 for (v1, v2) in someMatch]
        classD = [v2 for (v1, v2) in someMatch]
        classB = [v for v in nodesG1 if(not v in classA)]
        classC = [v for v in nodesG2 if(not v in classD)]
        # new labels of vertices for the alignment
        k = 0
        for v in classA:
            theMapToParentG1[v] = k
            k = k + 1
        for v in classB:
            theMapToParentG1[v] = k
            k = k + 1
        for v in classC:
            theMapToParentG2[v] = k
            k = k + 1
        # build candidate existence dictionary by "concatenating" alignment columns
        testExistence = dict()
        theLeaves = list(list(existenceG1.values())[0].keys())
        forwardMatch = {v1:v2 for (v1, v2) in someMatch}
        for v in classA:
            vP = theMapToParentG1[v]
            testExistence[vP] = dict()
            for leaf in theLeaves:
                testExistence[vP][leaf] = "."
                if(leaf in containedG1):
                    testExistence[vP][leaf] = deepcopy(existenceG1[v][leaf])
                if(leaf in containedG2):
                    testExistence[vP][leaf] = deepcopy(existenceG2[forwardMatch[v]][leaf])
        for v in classB:
            vP = theMapToParentG1[v]
            testExistence[vP] = dict()
            for leaf in theLeaves:
                testExistence[vP][leaf] = "."
                if(leaf in containedG1):
                    testExistence[vP][leaf] = deepcopy(existenceG1[v][leaf])
                if(leaf in containedG2):
                    testExistence[vP][leaf] = "-"
        for v in classC:
            vP = theMapToParentG2[v]
            testExistence[vP] = dict()
            for leaf in theLeaves:
                testExistence[vP][leaf] = "."
                if(leaf in containedG1):
                    testExistence[vP][leaf] = "-"
                if(leaf in containedG2):
                    testExistence[vP][leaf] = deepcopy(existenceG2[v][leaf])
        # get score from sum of scores
        squareLeaves = list(combinations(containedG1 + containedG2, r = 2))
        for vP in list(testExistence.keys()):
            # reinitialize sum of scores
            sumColumn = 0
            # do sum of scores
            for (L1, L2) in squareLeaves:
                if(not "-" in [testExistence[vP][L1], testExistence[vP][L2]]):
                    if(testExistence[vP][L1] == testExistence[vP][L2]):
                        # match
                        sumColumn = sumColumn + scoreSameLabel
                    else:
                        # missmatch
                        sumColumn = sumColumn + scoreDiffLabel
                else:
                    if(testExistence[vP][L1] == "-" and testExistence[vP][L2] == "-"):
                        # double gap
                        sumColumn = sumColumn + scoreGapGap
                    else:
                        # insertion / deletion
                        sumColumn = sumColumn + scoreLabelGap
            # add to overall score
            theScore = theScore + sumColumn
    # end of function
    return(theScore)


# function: kronecker delta required by sspk for "symb" and "nsymb" ------------
def kroneckerDeltaOne(label1, label2):
    # local variables
    resultSame = 1
    resultDiff = 0
    # end of function
    if(label1 == label2):
        return(resultSame)
    else:
        return(resultDiff)


# function: kronecker delta required by sspk for "mix" -------------------------
def kroneckerDeltaTwo(label1, label2, weight1, weight2):
    # local variables
    resultSame = 1
    resultDiff = 0
    # end of function
    if((label1 == label2) and (weight1 == weight2)):
        return(resultSame)
    else:
        return(resultDiff)


# function: get pairwise alignment and alignment score  ------------------------
def pairwiseAlignment(someG1, someG2,
                      existenceG1, existenceG2,
                      containedG1, containedG2,
                      graphType,
                      score = "match",
                      ambiguous1 = [], ambiguous2 = [],
                      vLabels = True,
                      heuristic = False,
                      oneSubIso = True,
                      subIsoLabeled = True,
                      oneMaxColoredMatch = True,
                      printProgressMCS = False,
                      ambiguousPairsCheck = False):
    # local variables
    minROFI = 0
    alignmentScore = 0
    match = []
    valsROFI = []
    possibleMatches = []
    isMatchAlsoSubIso = False
    isMatchAlsoMaxColored = False
    # get MCS
    if(graphType == "undirected"):
        possibleMatches, isMatchAlsoSubIso, isMatchAlsoMaxColored = undirMatchMCS(someG1, someG2,
                                                                                  existenceG1, existenceG2,
                                                                                  containedG1, containedG2,
                                                                                  ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                                                  score = score,
                                                                                  vLabels = vLabels,
                                                                                  heuristic = heuristic,
                                                                                  oneSubIso = oneSubIso,
                                                                                  subIsoLabeled = subIsoLabeled,
                                                                                  oneMaxColoredMatch = oneMaxColoredMatch,
                                                                                  printProgressMCS = printProgressMCS,
                                                                                  ambiguousPairsCheck = ambiguousPairsCheck)
    if(graphType == "directed"):
        possibleMatches, isMatchAlsoSubIso, isMatchAlsoMaxColored = dirMatchMCS(someG1, someG2,
                                                                                existenceG1, existenceG2,
                                                                                containedG1, containedG2,
                                                                                ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                                                score = score,
                                                                                vLabels = vLabels,
                                                                                heuristic = heuristic,
                                                                                oneSubIso = oneSubIso,
                                                                                subIsoLabeled = subIsoLabeled,
                                                                                oneMaxColoredMatch = oneMaxColoredMatch,
                                                                                printProgressMCS = printProgressMCS,
                                                                                ambiguousPairsCheck = ambiguousPairsCheck)
    # check if MCS is empty (possible when considering vertex labels)
    if(len(possibleMatches) > 0):
        valsROFI = [eachROFI for (eachMatch, eachROFI) in possibleMatches]
        # select match minimizing ROFI or being (sub)graph isomorphism
        if("SubIso" in valsROFI):
            match = [eachMatch for (eachMatch, eachROFI) in possibleMatches if(eachROFI == "SubIso")][0]
        else:
            minROFI = min(valsROFI)
            match = [eachMatch for (eachMatch, eachROFI) in possibleMatches if(eachROFI == minROFI)][0]
    else:
        match = []
    # get final score
    alignmentScore = matchScore(someG1, someG2, match,
                                existenceG1, existenceG2,
                                containedG1, containedG2,
                                score = score)
    # end of function
    return(alignmentScore, match)


# function: get ROFI for evaluation of the heuristic ---------------------------
def getROFI(someG1, someG2, someMatch):
    # local variables
    theROFI = 0
    nodesMCS1 = []
    nodesMCS2 = []
    nodeKs = dict()
    edgeKs = dict()
    sspkG1 = None
    sspkG2 = None
    kernelMatrix = None
    # determine sub-kernels tu be used
    nodeKs = {"symb": kroneckerDeltaOne, "nsymb": kroneckerDeltaOne, "mix": kroneckerDeltaTwo}
    edgeKs = {"symb": kroneckerDeltaOne, "nsymb": kroneckerDeltaOne, "mix": kroneckerDeltaTwo}
    # extend and re-condense graphs for sspk evaluation
    sspkG1 = deepcopy(condensedLabelToStr(extendedLabel(deepcopy(someG1))))
    sspkG2 = deepcopy(condensedLabelToStr(extendedLabel(deepcopy(someG2))))
    # get matched nodes
    nodesMCS1 = [a for (a, b) in someMatch]
    nodesMCS2 = [b for (a, b) in someMatch]
    # get complements of mcs
    sspkG1.remove_nodes_from(nodesMCS1)
    sspkG2.remove_nodes_from(nodesMCS2)
    # evaluate sspk
    kernelMatrix = sspk.structuralspkernel(sspkG1, sspkG2, parallel = None, verbose = False,
                                           node_label = "nodeLabelStr", edge_label = "edgeLabelStr",
                                           node_kernels = nodeKs, edge_kernels = edgeKs)[0]
    # get ROFI from kernel matrix
    theROFI = kernelMatrix[0][1]
    # end of function
    return(theROFI)


# function: tesi if the given match is a (sub)graph isomorphism ----------------
def isSubIso(someG1, someG2, someMatch):
    # check labels of matching vertices (can be different if not preserved by alignment)
    for (n1, n2) in someMatch:
        if(not someG1.nodes[n1]["nodeLabel"] == someG2.nodes[n2]["nodeLabel"]):
            return(False)
    # end of function
    return(True)


# function: check if unmatched vertices have disjoint sets of labels -----------
def isMaxColoredMatch(someG1, someG2, someMatch):
    # local variables
    nodesMCS1 = []
    nodesMCS2 = []
    labelIntersection = []
    labels1 = set()
    labels2 = set()
    strG1 = None
    strG2 = None
    # extend and re-condense labels into string
    strG1 = deepcopy(condensedLabelToStr(extendedLabel(deepcopy(someG1))))
    strG2 = deepcopy(condensedLabelToStr(extendedLabel(deepcopy(someG2))))
    # get matched nodes
    nodesMCS1 = [a for (a, b) in someMatch]
    nodesMCS2 = [b for (a, b) in someMatch]
    # get comolements of mcs
    strG1.remove_nodes_from(nodesMCS1)
    strG2.remove_nodes_from(nodesMCS2)
    # get label sets
    labels1 = set([strG1.nodes[v]["nodeLabelStr"] for v in list(strG1.nodes())])
    labels2 = set([strG2.nodes[v]["nodeLabelStr"] for v in list(strG2.nodes())])
    # get intersection
    labelIntersection = list(labels1.intersection(labels2))
    # end of function
    return(len(labelIntersection) == 0)


# function: determine ambiguous pairs of a given alignment ---------------------
def getAmbiguousPairs(someAlignment, someExistence, someContained):
    # local variables
    ambiguous = True
    theAmbiguous = []
    # get vertices in alignment
    vertices = list(someAlignment.nodes())
    # get ambiguous pairs
    pairs = combinations(vertices, r = 2)
    for (u, v) in pairs:
        # reinitialize
        ambiguous = True
        # compare vector entries
        for eachLeaf in someContained:
            if(not "-" in [someExistence[u][eachLeaf], someExistence[v][eachLeaf]]):
                ambiguous = False
                break
        # save if ambiguous
        if(ambiguous):
            theAmbiguous.append((u, v))
    # end of function
    return(theAmbiguous)


# functions - UNDIRECTED MCS ###################################################


# function: recursive MATCH for undir MCS search -------------------------------
def undirMatchMCS(someG1, someG2,
                  existenceG1, existenceG2,
                  containedG1, containedG2,
                  someMatch = [], allMatches = [],
                  ambiguous1 = [], ambiguous2 = [],
                  score = "match",
                  vLabels = True,
                  heuristic = False,
                  oneSubIso = True,
                  subIsoLabeled = True,
                  oneMaxColoredMatch = True,
                  printProgressMCS = True,
                  ambiguousPairsCheck = False,
                  totOrder = dict()):
    # local variables
    minROFI = 0
    progress = 0
    expOrder = 0
    scoreNewMatch = 0
    scoreOldMatch = 0
    candidateROFI = 0
    valsROFI = []
    newMatch = []
    currMatch1 = []
    currMatch2 = []
    candidatePairs = []
    forMatch = dict()
    invMatch = dict()
    allMatchesSet = set()
    ansSiFy = False
    ansSeFy = False
    foundSubIso = False
    progressReport = False
    foundMaxColoredMatch = False
    # get expected order for decision making
    expOrder = min([someG1.order(), someG2.order()])
    # define total order if not yet defined
    if(len(totOrder) == 0):
        totOrder = {v: i for (i, v) in enumerate(list(someG2.nodes()))}
        progressReport = True
    # test initial alignment and improvement in alignment
    if(len(allMatches) == 0):
        if(len(someMatch) > 0):
            # save match and its ROFI or "SubIso" if match is (sub)graph isomorphism
            if(len(someMatch) < expOrder):
                allMatches = [(someMatch, getROFI(someG1, someG2, someMatch))]
                if(oneMaxColoredMatch):
                    if(isMaxColoredMatch(someG1, someG2, someMatch)):
                        foundMaxColoredMatch = True
                        return(allMatches, foundSubIso, foundMaxColoredMatch)
            else:
                allMatches = [(someMatch, "SubIso")]
                if(oneSubIso):
                    if(vLabels):
                        foundSubIso = True
                        foundMaxColoredMatch = True
                    else:
                        if(subIsoLabeled):
                            if(isSubIso(someG1, someG2, someMatch)):
                                foundSubIso = True
                                foundMaxColoredMatch = True
                        else:
                            foundSubIso = True
                            if(isSubIso(someG1, someG2, someMatch)):
                                foundMaxColoredMatch = True
    else:
        # pick score based on arguments
        scoreNewMatch = matchScore(someG1, someG2, someMatch, existenceG1, existenceG2, containedG1, containedG2, score = score)
        scoreOldMatch = matchScore(someG1, someG2, allMatches[0][0], existenceG1, existenceG2, containedG1, containedG2, score = score)
        # save to MCS list if gets the same score of alignment
        if(scoreNewMatch == scoreOldMatch):
            allMatchesSet = [set(eachMatch) for (eachMatch, eachROFI) in allMatches]
            if(not set(someMatch) in allMatchesSet):
                # append match and its ROFI
                if(len(someMatch) < expOrder):
                    allMatches = allMatches + [(someMatch, getROFI(someG1, someG2, someMatch))]
                    if(oneMaxColoredMatch):
                        if(isMaxColoredMatch(someG1, someG2, someMatch)):
                            foundMaxColoredMatch = True
                            return(allMatches, foundSubIso, foundMaxColoredMatch)
                else:
                    allMatches = allMatches + [(someMatch, "SubIso")]
                    if(oneSubIso):
                        if(vLabels):
                            foundSubIso = True
                            foundMaxColoredMatch = True
                        else:
                            if(subIsoLabeled):
                                if(isSubIso(someG1, someG2, someMatch)):
                                    foundSubIso = True
                                    foundMaxColoredMatch = True
                            else:
                                foundSubIso = True
                                if(isSubIso(someG1, someG2, someMatch)):
                                    foundMaxColoredMatch = True
        # overwrite MCS list if there is inprovement in alignment
        if(scoreNewMatch > scoreOldMatch):
            if(len(someMatch) < expOrder):
                allMatches = [(someMatch, getROFI(someG1, someG2, someMatch))]
                if(oneMaxColoredMatch):
                    if(isMaxColoredMatch(someG1, someG2, someMatch)):
                        foundMaxColoredMatch = True
                        return(allMatches, foundSubIso, foundMaxColoredMatch)
            else:
                allMatches = [(someMatch, "SubIso")]
                if(oneSubIso):
                    if(vLabels):
                        foundSubIso = True
                        foundMaxColoredMatch = True
                    else:
                        if(subIsoLabeled):
                            if(isSubIso(someG1, someG2, someMatch)):
                                foundSubIso = True
                                foundMaxColoredMatch = True
                        else:
                            foundSubIso = True
                            if(isSubIso(someG1, someG2, someMatch)):
                                foundMaxColoredMatch = True
        # apply heuristic if it is the case
        if((scoreNewMatch < scoreOldMatch) and (len(someMatch) < expOrder) and heuristic):
            valsROFI = [eachROFI for (eachMatch, eachROFI) in allMatches]
            if(not "SubIso" in valsROFI):
                minROFI = min(valsROFI)
                candidateROFI = getROFI(someG1, someG2, someMatch)
                if(not candidateROFI >= minROFI):
                    return(allMatches, foundSubIso, foundMaxColoredMatch)
    # pre-evaluate available pairs
    if(len(someMatch) < expOrder):
        # generate auxiliary structures
        currMatch1 = [x for (x, y) in someMatch]
        currMatch2 = [y for (x, y) in someMatch]
        forMatch = {x:y for (x, y) in someMatch}
        invMatch = {y:x for (x, y) in someMatch}
        # get candidate pairs (if any)
        candidatePairs = undirCandidatesMCS(someMatch, currMatch1, currMatch2, someG1, someG2, totOrder)
        # evaluate candidate pairs
        for (n1, n2) in candidatePairs:
            # print progress only in first call
            if(progressReport and printProgressMCS):
                progress = progress + 1
                printProgress(round(progress*100/len(candidatePairs), 2), progressIn = "in case", reportCase = False)
            # evaluate sintactic feasibility
            ansSiFy = undirSintacticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someG2,
                                                   ambiguousPairsCheck, ambiguous1, ambiguous2)
            if(ansSiFy):
                # evaluate semantic feasibility
                ansSeFy = undirSemanticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, someG1, someG2)
                if(ansSeFy):
                    # DFS over feasible pairs
                    newMatch = someMatch + [(n1, n2)]
                    allMatches, foundSubIso, foundMaxColoredMatch = undirMatchMCS(someG1, someG2,
                                                                                  existenceG1, existenceG2,
                                                                                  containedG1, containedG2,
                                                                                  newMatch, allMatches,
                                                                                  ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                                                  score = score,
                                                                                  vLabels = vLabels,
                                                                                  heuristic = heuristic,
                                                                                  oneSubIso = oneSubIso,
                                                                                  subIsoLabeled = subIsoLabeled,
                                                                                  oneMaxColoredMatch = oneMaxColoredMatch,
                                                                                  ambiguousPairsCheck = ambiguousPairsCheck,
                                                                                  totOrder = totOrder)
                    # stop serach if one (sub)graph isomorphism was found (when requested)
                    if(oneSubIso and foundSubIso):
                        break
                    # stop search if one maximal colored match was found (when requested)
                    if(oneMaxColoredMatch and foundMaxColoredMatch):
                        break
    # end of function
    return(allMatches, foundSubIso, foundMaxColoredMatch)


# function: get candidate pairs for undir MCS search ---------------------------
def undirCandidatesMCS(someMatch, currMatch1, currMatch2, someG1, someG2, totOrder):
    # local variables
    maxMatchedIndex = 0
    P = []
    valid1 = []
    valid2 = []
    # get candidate pairs
    valid1 = [x for x in list(someG1.nodes()) if(not x in currMatch1)]
    valid2 = [y for y in list(someG2.nodes()) if(not y in currMatch2)]
    P = list(product(valid1, valid2))
    # get candidates preserving order (if no previous match just take everything)
    maxMatchedIndex = 0
    if(len(someMatch) > 0):
        maxMatchedIndex = max([totOrder[n2] for (n1, n2) in someMatch])
        P = [(x, y) for (x, y) in P if(totOrder[y] > maxMatchedIndex)]
    # end of function
    return(P)


# function: evaluate the sintactic feasability of mapping n1 to n2 -------------
def undirSintacticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someG2,
                                 ambiguousCheck, ambiguousG1, ambiguousG2):
    # local variables
    neigh1 = []
    neigh2 = []
    ambNeigh1 = []
    ambNeigh2 = []
    matchNeigh1 = []
    matchNeigh2 = []
    # get neighbors of n1 and n2
    neigh1 = list(someG1.neighbors(n1))
    neigh2 = list(someG2.neighbors(n2))
    # loop-consistency-test
    if((n1 in neigh1) and (not n2 in neigh2)):
        return(False)
    if((not n1 in neigh1) and (n2 in neigh2)):
        return(False)
    # look ahead 0: consistency of neighbors in match
    matchNeigh1 = [x for x in neigh1 if(x in currMatch1)]
    matchNeigh2 = [y for y in neigh2 if(y in currMatch2)]
    # get ambiguous neighbors if requested
    if(ambiguousCheck):
        ambNeigh1 = list(set([u for (u, v) in ambiguousG1 if(v == n1)] + [v for (u, v) in ambiguousG1 if(u == n1)]))
        ambNeigh2 = list(set([u for (u, v) in ambiguousG2 if(v == n2)] + [v for (u, v) in ambiguousG2 if(u == n2)]))
    # compare neighborhoods
    for v1 in matchNeigh1:
        # check if either true or ambiguous neighbor
        if(not forMatch[v1] in (matchNeigh2 + ambNeigh2)):
            return(False)
    for v2 in matchNeigh2:
        # check if either true or ambiguous neighbor
        if(not invMatch[v2] in (matchNeigh1 + ambNeigh1)):
            return(False)
    # end of function
    return(True)


# function: evaluate the semantic feasability of mapping n1 to n2 --------------
def undirSemanticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, someG1, someG2, vLabels = True):
    # local variables
    neigh1 = []
    neigh2 = []
    matchNeigh1 = []
    # compare vertex-labels
    if(vLabels):
        if(not someG1.nodes[n1]["nodeLabel"] == someG2.nodes[n2]["nodeLabel"]):
            return(False)
    # get neighborhoods
    neigh1 = list(someG1.neighbors(n1))
    # compare loop-label (if any)
    if(n1 in neigh1):
        if(not someG1[n1][n1]["edgeLabel"] == someG2[n2][n2]["edgeLabel"]):
            return(False)
    # compare edge-labels of true-edges and ignore ambiguous neighbors
    neigh2 = list(someG2.neighbors(n2))
    matchNeigh1 = [v for v in neigh1 if(v in currMatch1)]
    matchNeigh2 = [v for v in neigh2 if(v in currMatch2)]
    for v in matchNeigh1:
        # only true or ambiguous neighbors at this point
        if(forMatch[v] in matchNeigh2):
            if(not someG1[n1][v]["edgeLabel"] == someG2[n2][forMatch[v]]["edgeLabel"]):
                return(False)
    # end of function
    return(True)


# functions - DIRECTED MCS #####################################################


# function: recursive MATCH for dir MCS search ---------------------------------
def dirMatchMCS(someG1, someG2,
                existenceG1, existenceG2,
                containedG1, containedG2,
                someMatch = [], allMatches = [],
                ambiguous1 = [], ambiguous2 = [],
                score = "match",
                vLabels = True,
                heuristic = False,
                oneSubIso = True,
                subIsoLabeled = True,
                oneMaxColoredMatch = True,
                printProgressMCS = True,
                ambiguousPairsCheck = False,
                totOrder = dict()):
    # local variables
    minROFI = 0
    expOrder = 0
    progress = 0
    scoreNewMatch = 0
    scoreOldMatch = 0
    candidateROFI = 0
    valsROFI = []
    newMatch = []
    currMatch1 = []
    currMatch2 = []
    candidatePairs = []
    forMatch = dict()
    invMatch = dict()
    allMatchesSet = set()
    ansSiFy = False
    ansSeFy = False
    foundSubIso = False
    progressReport = False
    foundMaxColoredMatch = False
    # get expected order for decision making
    expOrder = min([someG1.order(), someG2.order()])
    # define total order if not yet defined
    if(len(totOrder) == 0):
        totOrder = {v: i for (i, v) in enumerate(list(someG2.nodes()))}
        progressReport = True
    # test initial alignment and improvement in alignment
    if(len(allMatches) == 0):
        if(len(someMatch) > 0):
            # save match and its ROFI or "SubIso" if match is (sub)graph isomorphism
            if(len(someMatch) < expOrder):
                allMatches = [(someMatch, getROFI(someG1, someG2, someMatch))]
                if(oneMaxColoredMatch):
                    if(isMaxColoredMatch(someG1, someG2, someMatch)):
                        foundMaxColoredMatch = True
                        return(allMatches, foundSubIso, foundMaxColoredMatch)
            else:
                allMatches = [(someMatch, "SubIso")]
                if(oneSubIso):
                    if(vLabels):
                        foundSubIso = True
                        foundMaxColoredMatch = True
                    else:
                        if(subIsoLabeled):
                            if(isSubIso(someG1, someG2, someMatch)):
                                foundSubIso = True
                                foundMaxColoredMatch = True
                        else:
                            foundSubIso = True
                            if(isSubIso(someG1, someG2, someMatch)):
                                foundMaxColoredMatch = True
    else:
        # pick score based on arguments
        scoreNewMatch = matchScore(someG1, someG2, someMatch, existenceG1, existenceG2, containedG1, containedG2, score = score)
        scoreOldMatch = matchScore(someG1, someG2, allMatches[0][0], existenceG1, existenceG2, containedG1, containedG2, score = score)
        # save to MCS list if gets the same score of alignment
        if(scoreNewMatch == scoreOldMatch):
            allMatchesSet = [set(eachMatch) for (eachMatch, eachROFI) in allMatches]
            if(not set(someMatch) in allMatchesSet):
                # append match and its ROFI
                if(len(someMatch) < expOrder):
                    allMatches = allMatches + [(someMatch, getROFI(someG1, someG2, someMatch))]
                    if(oneMaxColoredMatch):
                        if(isMaxColoredMatch(someG1, someG2, someMatch)):
                            foundMaxColoredMatch = True
                            return(allMatches, foundSubIso, foundMaxColoredMatch)
                else:
                    allMatches = allMatches + [(someMatch, "SubIso")]
                    if(oneSubIso):
                        if(vLabels):
                            foundSubIso = True
                            foundMaxColoredMatch = True
                        else:
                            if(subIsoLabeled):
                                if(isSubIso(someG1, someG2, someMatch)):
                                    foundSubIso = True
                                    foundMaxColoredMatch = True
                            else:
                                foundSubIso = True
                                if(isSubIso(someG1, someG2, someMatch)):
                                    foundMaxColoredMatch = True
        # overwrite MCS list if there is inprovement in alignment
        if(scoreNewMatch > scoreOldMatch):
            if(len(someMatch) < expOrder):
                allMatches = [(someMatch, getROFI(someG1, someG2, someMatch))]
                if(oneMaxColoredMatch):
                    if(isMaxColoredMatch(someG1, someG2, someMatch)):
                        foundMaxColoredMatch = True
                        return(allMatches, foundSubIso, foundMaxColoredMatch)
            else:
                allMatches = [(someMatch, "SubIso")]
                if(oneSubIso):
                    if(vLabels):
                        foundSubIso = True
                        foundMaxColoredMatch = True
                    else:
                        if(subIsoLabeled):
                            if(isSubIso(someG1, someG2, someMatch)):
                                foundSubIso = True
                                foundMaxColoredMatch = True
                        else:
                            foundSubIso = True
                            if(isSubIso(someG1, someG2, someMatch)):
                                foundMaxColoredMatch = True
        # apply heuristic if it is the case
        if((scoreNewMatch < scoreOldMatch) and (len(someMatch) < expOrder) and heuristic):
            valsROFI = [eachROFI for (eachMatch, eachROFI) in allMatches]
            if(not "SubIso" in valsROFI):
                minROFI = min(valsROFI)
                candidateROFI = getROFI(someG1, someG2, someMatch)
                if(not candidateROFI >= minROFI):
                    return(allMatches, foundSubIso, foundMaxColoredMatch)
    # pre-evaluate available pairs
    if(len(someMatch) < expOrder):
        # generate auxiliary structures
        currMatch1 = [x for (x, y) in someMatch]
        currMatch2 = [y for (x, y) in someMatch]
        forMatch = {x:y for (x, y) in someMatch}
        invMatch = {y:x for (x, y) in someMatch}
        # get candidate pairs (if any)
        candidatePairs = dirCandidatesMCS(someMatch, currMatch1, currMatch2, someG1, someG2, totOrder)
        # evaluate candidate pairs
        for (n1, n2) in candidatePairs:
            # print progress only in first call
            if(progressReport and printProgressMCS):
                progress = progress + 1
                printProgress(round(progress*100/len(candidatePairs), 2), progressIn = "in case", reportCase = False)
            # evaluate sintactic feasibility
            ansSiFy = dirSintacticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someG2,
                                                 ambiguousPairsCheck, ambiguous1, ambiguous2)
            if(ansSiFy):
                # evaluate semantic feasibility
                ansSeFy = dirSemanticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, someG1, someG2, vLabels = vLabels)
                if(ansSeFy):
                    # DFS over feasible pairs
                    newMatch = someMatch + [(n1, n2)]
                    allMatches, foundSubIso, foundMaxColoredMatch = dirMatchMCS(someG1, someG2,
                                                                                existenceG1, existenceG2,
                                                                                containedG1, containedG2,
                                                                                newMatch, allMatches,
                                                                                ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                                                score = score,
                                                                                vLabels = vLabels,
                                                                                heuristic = heuristic,
                                                                                oneSubIso = oneSubIso,
                                                                                subIsoLabeled = subIsoLabeled,
                                                                                oneMaxColoredMatch = oneMaxColoredMatch,
                                                                                ambiguousPairsCheck = ambiguousPairsCheck,
                                                                                totOrder = totOrder)
                    # stop serach is one (sub)graph isomorphism was found (only if oneSubIso option is requested)
                    if(oneSubIso and foundSubIso):
                        break
                    # stop search if one maximal colored match was found (when requested)
                    if(oneMaxColoredMatch and foundMaxColoredMatch):
                        break
    # end of function
    return(allMatches, foundSubIso, foundMaxColoredMatch)


# function: get candidate pairs for dir MCS search ------------------------------
def dirCandidatesMCS(someMatch, currMatch1, currMatch2, someG1, someG2, totOrder):
    # local variables
    maxMatchedIndex = 0
    P = []
    valid1 = []
    valid2 = []
    # alternatively try pairing all unpaired vertices
    valid1 = [x for x in list(someG1.nodes()) if(not x in currMatch1)]
    valid2 = [y for y in list(someG2.nodes()) if(not y in currMatch2)]
    P = list(product(valid1, valid2))
    # get candidates preserving order (if no previous match just take everything)
    maxMatchedIndex = 0
    if(len(someMatch) > 0):
        maxMatchedIndex = max([totOrder[n2] for (n1, n2) in someMatch])
        P = [(x, y) for (x, y) in P if(totOrder[y] > maxMatchedIndex)]
    # end of function
    return(P)


# function: evaluate the sintactic feasability of mapping n1 to n2 -------------
def dirSintacticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someG2,
                               ambiguousCheck, ambiguousG1, ambiguousG2):
    # local variables
    inNeigh1 = []
    inNeigh2 = []
    outNeigh1 = []
    outNeigh2 = []
    ambNeigh1 = []
    ambNeigh2 = []
    inMatchNeigh1 = []
    inMatchNeigh2 = []
    outMatchNeigh1 = []
    outMatchNeigh2 = []
    # get neighbors of n1 and n2
    inNeigh1 = list(someG1.predecessors(n1))
    inNeigh2 = list(someG2.predecessors(n2))
    outNeigh1 = list(someG1.neighbors(n1))
    outNeigh2 = list(someG2.neighbors(n2))
    # get ambiguous neighbors if requested
    # loop-consistency-test
    if((n1 in inNeigh1) and (not n2 in inNeigh2)):
        return(False)
    if((not n1 in inNeigh1) and (n2 in inNeigh2)):
        return(False)
    # get ambiguous neighbors if requested
    if(ambiguousCheck):
        ambNeigh1 = list(set([u for (u, v) in ambiguousG1 if(v == n1)] + [v for (u, v) in ambiguousG1 if(u == n1)]))
        ambNeigh2 = list(set([u for (u, v) in ambiguousG2 if(v == n2)] + [v for (u, v) in ambiguousG2 if(u == n2)]))
    # look ahead 0: consistency of neighbors in match
    inMatchNeigh1 = [a1 for a1 in inNeigh1 if(a1 in currMatch1)]
    inMatchNeigh2 = [a2 for a2 in inNeigh2 if(a2 in currMatch2)]
    for a1 in inMatchNeigh1:
        if(not forMatch[a1] in (inMatchNeigh2 + ambNeigh2)):
            return(False)
    for a2 in inMatchNeigh2:
        if(not invMatch[a2] in (inMatchNeigh1 + ambNeigh1)):
            return(False)
    outMatchNeigh1 = [b1 for b1 in outNeigh1 if(b1 in currMatch1)]
    outMatchNeigh2 = [b2 for b2 in outNeigh2 if(b2 in currMatch2)]
    for b1 in outMatchNeigh1:
        if(not forMatch[b1] in (outMatchNeigh2 + ambNeigh2)):
            return(False)
    for b2 in outMatchNeigh2:
        if(not invMatch[b2] in (outMatchNeigh1 + ambNeigh1)):
            return(False)
    # end of function
    return(True)


# function: evaluate the semantic feasability of mapping n1 to n2 --------------
def dirSemanticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, someG1, someG2, vLabels = True):
    # local variables
    inNeigh1 = []
    inNeigh2 = []
    outNeigh1 = []
    outNeigh2 = []
    inMatchNeigh1 = []
    inMatchNeigh2 = []
    outMatchNeigh1 = []
    outMatchNeigh2 = []
    # compare vertex-labels
    if(vLabels):
        if(not someG1.nodes[n1]["nodeLabel"] == someG2.nodes[n2]["nodeLabel"]):
            return(False)
    # get neighborhoods
    inNeigh1 = list(someG1.predecessors(n1))
    outNeigh1 = list(someG1.neighbors(n1))
    # compare loop-label (if any)
    if(n1 in inNeigh1):
        if(not someG1[n1][n1]["edgeLabel"] == someG2[n2][n2]["edgeLabel"]):
            return(False)
    # compare edge-labels of true-edges and ignore ambiguous neighbors
    inNeigh2 = list(someG2.predecessors(n2))
    outNeigh2 = list(someG2.neighbors(n2))
    inMatchNeigh1 = [a for a in inNeigh1 if(a in currMatch1)]
    outMatchNeigh1 = [b for b in outNeigh1 if(b in currMatch1)]
    inMatchNeigh2 = [a for a in inNeigh2 if(a in currMatch2)]
    outMatchNeigh2 = [b for b in outNeigh2 if(b in currMatch2)]
    for a in inMatchNeigh1:
        # only true or ambiguous neighbors at this point
        if(forMatch[a] in inMatchNeigh2):
            if(not someG1[a][n1]["edgeLabel"] == someG2[forMatch[a]][n2]["edgeLabel"]):
                return(False)
    for b in outMatchNeigh1:
        # only true or ambiguous neighbors at this point
        if(forMatch[b] in outMatchNeigh2):
            if(not someG1[n1][b]["edgeLabel"] == someG2[n2][forMatch[b]]["edgeLabel"]):
                return(False)
    # end of function
    return(True)


# main #########################################################################


# initial message
print("\n")
print(">>> Progralign - Progralign Github Repository")


# task message
print("\n")
print("* Retrieving input file ...")


# retrieve input file
inputFile = open(inputFileName, "rb")
inputTuple = pkl.load(inputFile)
inputFile.close()


# get primal and input list of graphs
primal = deepcopy(inputTuple[0])
inputList = deepcopy(inputTuple[1])


# check non-empty list
if(not isinstance(inputList, list)):
    exit("\n >> Progralign: the file " + inputFileName + " does not contain a list.\n")
if(len(inputList) == 0):
    exit("\n >> Progralign: the file " + inputFileName + " contains an empty list.\n")


# task message
print("* Received " + str(len(inputList)) + " input graphs ...")


# task message
print("* Evaluating consistency of the input file ...")


# evaluate consistency of the input file
for eachGraph in inputList + [primal]:
    try:
        nx.is_directed(eachGraph)
    except:
        exit("\n >> Progralign: the file " + inputFileName + " contains an object which is not a networkx (di-)graph.\n")
if(not len(list(set([type(eachGraph) for eachGraph in inputList + [primal]]))) == 1):
    exit("\n >> Progralign: the file " + inputFileName + " contains both directed and undirected graphs. These must be of the same type.\n")
if(len(inputList) == 1):
    exit("\n >> Progralign: the file " + inputFileName + " contains only one (di-)graph. Alignment won't be computed.\n")
for eachGraph in inputList + [primal]:
    if(eachGraph.order() == 0):
        exit("\n >> Progralign: the file " + inputFileName + " contains a (di-)graph with no vertices. Alignment won't be computed.\n")
if(nx.is_directed(inputList[0])):
    theType = "directed"
else:
    theType = "undirected"


# define python's recursion limit based on the order of input graphs
scalationVal = 1.5
requiredLimit = max([eachGraph.order() for eachGraph in inputList + [primal]])
currentLimit = getrecursionlimit()
if(currentLimit < (scalationVal * requiredLimit)):
    setrecursionlimit(int(scalationVal * requiredLimit))


# task message
print("* Data is consistent. Indexing graphs ...")


# generate indices for the input graphs
indices = list(range(len(inputList)))
for i in indices:
    constructionDict[str(i)] = deepcopy(inputList[i])
constructionDict["Indices"] = deepcopy(indices)


# task message
print("* Evaluating Structural-Shortest-Path graph-kernel ...")


# evaluate graph-kernels
metaInputList = inputList + [primal]
inputListCondensed = [deepcopy(condensedLabelToStr(eachGraph)) for eachGraph in metaInputList]
nodeKernels = {"symb": kroneckerDeltaOne, "nsymb": kroneckerDeltaOne, "mix": kroneckerDeltaTwo}
edgeKernels = {"symb": kroneckerDeltaOne, "nsymb": kroneckerDeltaOne, "mix": kroneckerDeltaTwo}
kernelMatrixSSP = sspk.structuralspkernel(inputListCondensed, parallel = None, verbose = True,
                                          node_label = "nodeLabelStr", edge_label = "edgeLabelStr",
                                          node_kernels = nodeKernels, edge_kernels = edgeKernels)[0]
constructionDict["SimilarityMatrix"] = deepcopy(kernelMatrixSSP)


# task message
print("\n")
print("* Retrieving pairwise similarities and distances betwween input graphs ...")


# build similarity graph
clusteringGraphSSP = nx.Graph()
for (i, j) in list(combinations(indices, r = 2)):
    # build similarity
    if(guideTreeMeasure == "distance"):
        selfScoreG1 = kernelMatrixSSP[i][i]
        selfScoreG2 = kernelMatrixSSP[j][j]
        pairScoreG1G2 = kernelMatrixSSP[i][j]
        uvDistanceSSP = selfScoreG1 + selfScoreG2 - 2*pairScoreG1G2
        clusteringGraphSSP.add_edge(str(i), str(j), measure = uvDistanceSSP)
    if(guideTreeMeasure == "similarity"):
        uvSimilaritySSP = kernelMatrixSSP[i][j]
        clusteringGraphSSP.add_edge(str(i), str(j), measure = uvSimilaritySSP)


# task message
print("* Building guide tree ...")


# iterative construction of dendrograms
dendrogramSSP, guideTreeSSP, containedGraphsSSP = getGuideTree(clusteringGraphSSP, guideTreeMeasure)
constructionDict["GTMeasure"] = guideTreeMeasure
constructionDict["Dendrogram"] = dendrogramSSP
constructionDict["GuideTree"] = deepcopy(guideTreeSSP)
constructionDict["Contained"] = deepcopy(containedGraphsSSP)


# task message
print("* Performing progressive graph alignment based on guide-tree pruning:")


# prepare data for alignment
guideTree = deepcopy(guideTreeSSP)
# get leaves as current alignments
alignments = {h:deepcopy(condensedLabel(constructionDict[h])) for h in list(guideTree.nodes()) if(guideTree.degree(h) == 1)}
allLeaves = [h for h in list(guideTree.nodes()) if(guideTree.degree(h) == 1)]
allLeavesGraphs = deepcopy(alignments)
# get leaves values for existence dictionary and their depths
existence = dict()
for eachGraph in list(alignments.keys()):
    allScores[eachGraph] = 0
    mapToRoot[eachGraph] = dict()
    existence[eachGraph] = dict()
    graphLeaf = deepcopy(alignments[eachGraph])
    ambiguousPairs[eachGraph] = []
    for v in list(graphLeaf.nodes()):
        mapToRoot[eachGraph][v] = ""
        existence[eachGraph][v] = {str(i):"." for i in indices}
        existence[eachGraph][v][eachGraph] = deepcopy(graphLeaf.nodes[v]["nodeLabel"])
        # existence[eachGraph][v][eachGraph] = "x"
# get paths from leafs to root and depth of leaves
pathRoot = {h:nx.shortest_path(guideTree, source = h, target = dendrogramSSP) for h in list(alignments.keys())}
depthLeaves = {h:int(nx.shortest_path_length(guideTree, source = h, target = dendrogramSSP)) for h in list(alignments.keys())}
# get height of guide tree
heightGT = max([int(nx.shortest_path_length(guideTree, source = h, target = dendrogramSSP)) for h in list(alignments.keys())])


# start timer
initialTime = time.time()


# progressive graph alignment based on guide-tree pruning
mapToParent = dict()
counter = 0
while(len(list(alignments.keys())) > 1):
    # FIRST get vertices adjacent to two exactly two leaves
    cherries = []
    for v in list(guideTree.nodes()):
        leafNeighbors = [h for h in list(guideTree.neighbors(v)) if(guideTree.degree(h) == 1)]
        if(len(leafNeighbors) == 2):
            cherries.append(v)
    # THEN align leaf-neighbors of cherries and delete them turning cherries into leaves
    toRemove = []
    for cherry in cherries:
        # get the two leaf neighbors
        leafNeighbors = [h for h in list(guideTree.neighbors(cherry)) if(guideTree.degree(h) == 1)]
        (x, y) = tuple(sorted(leafNeighbors))
        # task message
        print("  * doing pairwise alignment of: ", x, "  with  ", y)
        # alignment for undirected graphs
        if(theType == "undirected"):
            # get MCS
            G1 = deepcopy(alignments[x])
            G2 = deepcopy(alignments[y])
            possibleMatches, isMatchAlsoSubIso, isMatchAlsoMaxColored = undirMatchMCS(G1, G2,
                                                                                      existence[x], existence[y],
                                                                                      containedGraphsSSP[x], containedGraphsSSP[y],
                                                                                      ambiguous1 = ambiguousPairs[x], ambiguous2 = ambiguousPairs[y],
                                                                                      score = requestedScore,
                                                                                      vLabels = respectVLabels,
                                                                                      heuristic = heuristicROFI,
                                                                                      oneSubIso = justOneSubIso,
                                                                                      subIsoLabeled = subIsoMustBeLabeled,
                                                                                      oneMaxColoredMatch = justOneMaximalColoredMatch,
                                                                                      ambiguousPairsCheck = ambiguousEdges)
            # check if MCS is empty (possible when considering vertex labels)
            if(len(possibleMatches) > 0):
                valsROFI = [eachROFI for (eachMatch, eachROFI) in possibleMatches]
                # select match minimizing ROFI or being (sub)graph isomorphism
                if("SubIso" in valsROFI):
                    match = [eachMatch for (eachMatch, eachROFI) in possibleMatches if(eachROFI == "SubIso")][0]
                else:
                    minROFI = min(valsROFI)
                    match = [eachMatch for (eachMatch, eachROFI) in possibleMatches if(eachROFI == minROFI)][0]
            else:
                match = []
            # build alignment graph and get final score
            resutls = buildAlignment(G1, G2, match,
                                     existence[x], existence[y],
                                     containedGraphsSSP[x], containedGraphsSSP[y])
            allScores[cherry] = matchScore(G1, G2, match,
                                           existence[x], existence[y],
                                           containedGraphsSSP[x], containedGraphsSSP[y],
                                           score = requestedScore)
            # store alignment graph and its attributes
            alignments[cherry] = deepcopy(resutls[0])
            existence[cherry] = deepcopy(resutls[1])
            mapToParent[x] = deepcopy(resutls[2])
            mapToParent[y] = deepcopy(resutls[3])
            ambiguousPairs[cherry] = getAmbiguousPairs(alignments[cherry], existence[cherry], containedGraphsSSP[cherry])
        # alignment for directed graphs
        if(theType == "directed"):
            # get MCS
            G1 = deepcopy(alignments[x])
            G2 = deepcopy(alignments[y])
            possibleMatches, isMatchAlsoSubIso, isMatchAlsoMaxColored = dirMatchMCS(G1, G2,
                                                                                    existence[x], existence[y],
                                                                                    containedGraphsSSP[x], containedGraphsSSP[y],
                                                                                    ambiguous1 = ambiguousPairs[x], ambiguous2 = ambiguousPairs[y],
                                                                                    score = requestedScore,
                                                                                    vLabels = respectVLabels,
                                                                                    heuristic = heuristicROFI,
                                                                                    oneSubIso = justOneSubIso,
                                                                                    subIsoLabeled = subIsoMustBeLabeled,
                                                                                    oneMaxColoredMatch = justOneMaximalColoredMatch,
                                                                                    ambiguousPairsCheck = ambiguousEdges)
            # check if MCS is empty (possible when considering vertex labels)
            if(len(possibleMatches) > 0):
                valsROFI = [eachROFI for (eachMatch, eachROFI) in possibleMatches]
                # select match minimizing ROFI or being (sub)graph isomorphism
                if("SubIso" in valsROFI):
                    match = [eachMatch for (eachMatch, eachROFI) in possibleMatches if(eachROFI == "SubIso")][0]
                else:
                    minROFI = min(valsROFI)
                    match = [eachMatch for (eachMatch, eachROFI) in possibleMatches if(eachROFI == minROFI)][0]
            else:
                match = []
            # build alignment graph and get final score
            resutls = buildAlignment(G1, G2, match,
                                     existence[x], existence[y],
                                     containedGraphsSSP[x], containedGraphsSSP[y])
            allScores[cherry] = matchScore(G1, G2, match,
                                           existence[x], existence[y],
                                           containedGraphsSSP[x], containedGraphsSSP[y],
                                           score = requestedScore)
            # store alignment graph and its attributes
            alignments[cherry] = deepcopy(resutls[0])
            existence[cherry] = deepcopy(resutls[1])
            mapToParent[x] = deepcopy(resutls[2])
            mapToParent[y] = deepcopy(resutls[3])
            ambiguousPairs[cherry] = getAmbiguousPairs(alignments[cherry], existence[cherry], containedGraphsSSP[cherry])
        # erease aligned leaves from guide-tree
        guideTree.remove_nodes_from([x, y])
        toRemove = toRemove + [x, y]
    # update alignments
    toRemove = list(set(toRemove))
    alignments = {name:alignments[name] for name in list(alignments.keys()) if(not name in toRemove)}
    # print progress
    counter = counter + 1
    print("  ", end = "")
    printProgress((counter*100)/heightGT, progressIn = "over the guide tree", reportCase = False, uTurn = False)


# end timer and save alignment time
finalTime = time.time()
alignmentTime = finalTime - initialTime
constructionDict["Time"] = alignmentTime


# task message
print("***** Concluded alignment")
print("***** Time taken by alignment [s]: ", alignmentTime)


# obtain vertex maps from input graphs to root
for eachLeaf in allLeaves:
    graphLeaf = deepcopy(allLeavesGraphs[eachLeaf])
    for v in list(graphLeaf.nodes()):
        tempMapV = mapToParent[eachLeaf][v]
        for eachInner in pathRoot[eachLeaf][1:-1]:
            tempMapV = mapToParent[eachInner][tempMapV]
        mapToRoot[eachLeaf][v] = tempMapV


# retrieve alignment and existence nested dictionary
finalAlignment = alignments[dendrogramSSP]
finalAlignment = deepcopy(extendedLabel(finalAlignment))
constructionDict["Alignment"] = deepcopy(finalAlignment)
constructionDict["Existence"] = deepcopy(existence)
finalScore = allScores[dendrogramSSP]
constructionDict["ScoreInfo"] = (requestedScore, finalScore, respectVLabels)
constructionDict["MapToRoot"] = deepcopy(mapToRoot)
constructionDict["AmbiguousEdges"] = deepcopy(ambiguousPairs)


# save extra data necessary for visualization
constructionDict["AllLeaves"] = deepcopy(allLeaves)
constructionDict["AllLeavesGraphs"] = deepcopy(allLeavesGraphs)


# task message
print("\n")
print("* Obtaining score-distances between primal and mutants ...")


# prepare data for evaluation of score-distance between mutants and primal
for eachLeaf in allLeaves:
    existencePrimalComp[eachLeaf] = deepcopy(existence[eachLeaf])
    graphLeaf = deepcopy(constructionDict[eachLeaf])
    for v in list(graphLeaf.nodes()):
        existencePrimalComp[eachLeaf][v]["primal"] = "."
# initialize existence for primal
condPrimal = deepcopy(condensedLabel(primal))
existencePrimalComp["primal"] = dict()
for v in list(condPrimal.nodes()):
    existencePrimalComp["primal"][v] = {str(i):"." for i in indices}
    existencePrimalComp["primal"][v]["primal"] = deepcopy(condPrimal.nodes[v]["nodeLabel"])
# get score primal-primal
if(basicSelfScoring):
    scorePrimalPrimal = condPrimal.order()
else:
    scorePrimalPrimal, matchPrimalPrimal = pairwiseAlignment(condPrimal, condPrimal,
                                                             existencePrimalComp["primal"], existencePrimalComp["primal"],
                                                             ["primal"], ["primal"],
                                                             theType,
                                                             score = requestedScore,
                                                             vLabels = respectVLabels,
                                                             heuristic = heuristicROFI,
                                                             oneSubIso = justOneSubIso,
                                                             subIsoLabeled = subIsoMustBeLabeled,
                                                             oneMaxColoredMatch = justOneMaximalColoredMatch)


# get score-distance between mutants and primal
leafCount = 0
scoreDistPM = 0
scoreDistancePM = dict()
scoreDistancePMVals = []
for eachLeaf in allLeaves:
    graphLeaf = deepcopy(condensedLabel(constructionDict[eachLeaf]))
    # get score primal-mutant
    scorePrimalMutant, matchPrimalMutant  = pairwiseAlignment(condPrimal, graphLeaf,
                                                              existencePrimalComp["primal"], existencePrimalComp[eachLeaf],
                                                              ["primal"], [eachLeaf],
                                                              theType,
                                                              score = requestedScore,
                                                              vLabels = respectVLabels,
                                                              heuristic = heuristicROFI,
                                                              oneSubIso = justOneSubIso,
                                                              subIsoLabeled = subIsoMustBeLabeled,
                                                              oneMaxColoredMatch = justOneMaximalColoredMatch)
    # get score mutant-mutant
    if(basicSelfScoring):
        scoreMutantMutant = graphLeaf.order()
    else:
        scoreMutantMutant, matchMutantMutant = pairwiseAlignment(graphLeaf, graphLeaf,
                                                                 existencePrimalComp[eachLeaf], existencePrimalComp[eachLeaf],
                                                                 [eachLeaf], [eachLeaf],
                                                                 theType,
                                                                 score = requestedScore,
                                                                 vLabels = respectVLabels,
                                                                 heuristic = heuristicROFI,
                                                                 subIsoLabeled = subIsoMustBeLabeled,
                                                                 oneMaxColoredMatch = justOneMaximalColoredMatch)
    # get distance score
    scoreDistPM = scorePrimalPrimal + scoreMutantMutant - (2*scorePrimalMutant)
    scoreDistancePM[eachLeaf] = (scorePrimalPrimal, scoreMutantMutant, scorePrimalMutant, scoreDistPM)
    scoreDistancePMVals.append(scoreDistPM)
    leafCount = leafCount + 1
    printProgress(round(leafCount*100/len(allLeaves), 2), reportCase = False)
constructionDict["ScoreDistancePM"] = deepcopy(scoreDistancePM)
constructionDict["ScoreDistancePMVals"] = deepcopy(scoreDistancePMVals)


# task message
print("\n")
print("* Obtaining kernel-distances between primal and mutants ...")


# get kernel-distance between mutants and primal
leafCount = 0
kernelDistPM = 0
kernelDistancePM = dict()
kernelDistancePMVals = []
scorePrimalPrimal = kernelMatrixSSP[len(allLeaves)][len(allLeaves)]
for eachLeaf in allLeaves:
    scorePrimalMutant = kernelMatrixSSP[int(eachLeaf)][len(allLeaves)]
    scoreMutantMutant = kernelMatrixSSP[int(eachLeaf)][int(eachLeaf)]
    kernelDistPM = scorePrimalPrimal + scoreMutantMutant - (2*scorePrimalMutant)
    kernelDistancePM[eachLeaf] = (scorePrimalPrimal, scoreMutantMutant, scorePrimalMutant, kernelDistPM)
    kernelDistancePMVals.append(kernelDistPM)
    leafCount = leafCount + 1
    printProgress(round(leafCount*100/len(allLeaves), 2), reportCase = False)
constructionDict["KernelDistancePM"] = deepcopy(kernelDistancePM)
constructionDict["KernelDistancePMVals"] = deepcopy(kernelDistancePMVals)


# task message
print("\n")
print("* Obtaining score-distances between pairs of mutants ...")


# get score-distance between pairs of mutants
leafCount = 0
scoreDistMM = 0
scoreDistanceMMVals = []
scoreDistanceMM = dict()
pairsOfMutants = list(combinations(allLeaves, r = 2))
for (eachLeaf1, eachLeaf2) in pairsOfMutants:
    # get graphs
    graphLeaf1 = deepcopy(condensedLabel(constructionDict[eachLeaf1]))
    graphLeaf2 = deepcopy(condensedLabel(constructionDict[eachLeaf2]))
    # get score first mutant
    if(basicSelfScoring):
        scoreMutant1 = graphLeaf1.order()
    else:
        scoreMutant1, matchMutant1  = pairwiseAlignment(graphLeaf1, graphLeaf1,
                                                        existence[eachLeaf1], existence[eachLeaf1],
                                                        [eachLeaf1], [eachLeaf1],
                                                        theType,
                                                        score = requestedScore,
                                                        vLabels = respectVLabels,
                                                        heuristic = heuristicROFI,
                                                        oneSubIso = justOneSubIso,
                                                        subIsoLabeled = subIsoMustBeLabeled,
                                                        oneMaxColoredMatch = justOneMaximalColoredMatch)
    # get score second mutant
    if(basicSelfScoring):
        scoreMutant2 = graphLeaf2.order()
    else:
        scoreMutant2, matchMutant2  = pairwiseAlignment(graphLeaf2, graphLeaf2,
                                                        existence[eachLeaf2], existence[eachLeaf2],
                                                        [eachLeaf2], [eachLeaf2],
                                                        theType,
                                                        score = requestedScore,
                                                        vLabels = respectVLabels,
                                                        heuristic = heuristicROFI,
                                                        oneSubIso = justOneSubIso,
                                                        subIsoLabeled = subIsoMustBeLabeled,
                                                        oneMaxColoredMatch = justOneMaximalColoredMatch)
    # get score between the two mutants
    scoreBetweenMutants, matchBetweenMutants = pairwiseAlignment(graphLeaf1, graphLeaf2,
                                                                 existence[eachLeaf1], existence[eachLeaf2],
                                                                 [eachLeaf1], [eachLeaf2],
                                                                 theType,
                                                                 score = requestedScore,
                                                                 vLabels = respectVLabels,
                                                                 heuristic = heuristicROFI,
                                                                 oneSubIso = justOneSubIso,
                                                                 subIsoLabeled = subIsoMustBeLabeled,
                                                                 oneMaxColoredMatch = justOneMaximalColoredMatch)
    # get distance score
    scoreDistMM = scoreMutant1 + scoreMutant2 - (2*scoreBetweenMutants)
    scoreDistanceMM[(eachLeaf1, eachLeaf2)] = (scoreMutant1, scoreMutant2, scoreBetweenMutants, scoreDistMM)
    scoreDistanceMMVals.append(scoreDistMM)
    leafCount = leafCount + 1
    printProgress(round(leafCount*100/len(pairsOfMutants), 2), reportCase = False)
constructionDict["ScoreDistanceMM"] = deepcopy(scoreDistanceMM)
constructionDict["ScoreDistanceMMVals"] = deepcopy(scoreDistanceMMVals)


# task message
print("\n")
print("* Obtaining kernel-distances between pairs of mutants ...")


# get kernel-distance between pairs of mutants
leafCount = 0
kernelDistMM = 0
kernelDistanceMMVals = []
kernelDistanceMM = dict()
for (eachLeaf1, eachLeaf2) in pairsOfMutants:
    scoreMutant1 = kernelMatrixSSP[int(eachLeaf1)][int(eachLeaf1)]
    scoreMutant2 = kernelMatrixSSP[int(eachLeaf2)][int(eachLeaf2)]
    scoreBetweenMutants = kernelMatrixSSP[int(eachLeaf1)][int(eachLeaf2)]
    kernelDistMM = scoreMutant1 + scoreMutant2 - (2*scoreBetweenMutants)
    kernelDistanceMM[(eachLeaf1, eachLeaf2)] = kernelDistMM
    kernelDistanceMMVals.append(kernelDistMM)
    leafCount = leafCount + 1
    printProgress(round(leafCount*100/len(allLeaves), 2), reportCase = False)
constructionDict["KernelDistanceMM"] = deepcopy(kernelDistanceMM)
constructionDict["KernelDistanceMMVals"] = deepcopy(kernelDistanceMMVals)


# task message
print("\n")
print("* Saving data ...")


# save results
outputFile = open(nameResultsPKL, "wb")
pkl.dump(constructionDict, outputFile)
outputFile.close()


# final message
print("\n")
print(">>> Finished")
print("\n")


# drawing section to be splitted into Vis2D -----------------------------------


# task message
print("\n")
print("*** Making plots of graphs ...")


# draw graphs
# requires: finalAlignment, Alignment, AllLeaves, AllLeavesGraphs, Indices, MapToRoot, dendrogramSSP, existence
vis1D = False
vis2D = False
vis3D = False
graphFigs3DInfo = []
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
    # drawing parameters
    minX = -1.15
    maxX = 1.15
    minY = -1.15
    maxY = 1.15
    propX = 12
    propY = 8
    apart = 0.25
    # get data
    finalAlignment = deepcopy(constructionDict["Alignment"])
    # get positions for vertices in the alignment
    fan = finalAlignment.order()
    valK = (1/sqrt(fan)) + apart
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
    # define color map based on values
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
        fig = plt.figure(figsize = (propX, propY))
        # plot alignment in background
        nx.draw_networkx(finalAlignment, with_labels = False, pos = posAlignment, node_size = 25,
                         node_color = "gainsboro", edge_color = "gainsboro", width = 0.25)
        # plot input graph
        nx.draw_networkx(graphLeaf, with_labels = True, labels = mapToRoot[eachLeaf],
                         pos = posLeaf, node_size = 200, nodelist = nodeList, node_color = nodeColor,
                         font_size = 8, edge_color = "k", width = 2)
        plt.ylabel("Input Graph " + eachLeaf, fontsize = 14, weight = "bold")
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
    fig = plt.figure(figsize = (propX, propY))
    # plot alignment
    nx.draw_networkx(finalAlignment, with_labels = True,
                     pos = posAlignment, node_size = 200, node_color = "silver",
                     font_size = 8, edge_color = "k", width = 2)
    plt.ylabel("Alignment", fontsize = 18, weight = "bold")
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
    fig = plt.figure()
    # draw final alignment
    nx.draw_networkx(finalAlignment, with_labels = True,
                     pos = posAlignment, node_size = 150,
                     font_size = 6, node_color = "silver",
                     edge_color = "k", width = 2)
    plt.ylabel("Alignment", fontsize = 11, weight = "bold")
    plt.xlim([minX, maxX])
    plt.ylim([minY, maxY])
    plt.tight_layout()
    # save page
    sanityCheckPDF.savefig(fig)
    # clear current figure
    plt.clf()
    # create figure
    fig = plt.figure()
    # build matrix
    nodePresence = dict()
    nodesAlignment = list(finalAlignment.nodes())
    nodesAlignment.sort()
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
    # define minor ticks
    ax = plt.gca()
    # set major tick positions
    ax.set_xticks(np.arange(len(nodesAlignment)))
    ax.set_yticks(np.arange(len(allLeaves)))
    # set major tick labels
    ax.set_xticklabels(nodesAlignment, fontsize = 4)
    ax.set_yticklabels(allLeaves, fontsize = 4)
    # set minor ticks
    ax.set_xticks(np.arange(-0.5, len(nodesAlignment), 1), minor = True)
    ax.set_yticks(np.arange(-0.5, len(allLeaves), 1), minor = True)
    # loop over data dimensions and create text annotations
    for i in range(len(allLeaves)):
        for j in range(len(nodesAlignment)):
            try:
                ax.text(j, i, str(int(maskedArray[i, j])), ha = "left", va = "bottom", color = "k", fontsize = 3.5)
            except:
                pass
    # set grid
    ax.grid(which = "minor", color = "k", linestyle = "-", linewidth = 1)
    # remove minor ticks
    ax.tick_params(which = "minor", bottom = False, left = False)
    # finish image
    plt.title("Matrix Representation of the Alignment\n", fontsize = 8, weight = "bold")
    plt.xlabel("\nVertices of the alignment", fontsize = 6.5, weight = "bold")
    plt.ylabel("Input Graphs\n", fontsize = 6.5, weight = "bold")
    # save page
    sanityCheckPDF.savefig(fig)
    # clear current figure
    plt.clf()
    # create figure
    fig = plt.figure()
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
    plt.title("Color code of vertex labels", fontsize = 8, weight = "bold")
    plt.tight_layout()
    # save page
    sanityCheckPDF.savefig(fig)
    # save pdf
    sanityCheckPDF.close()
    plt.close()


# task message
print("\n")
print("*** Making scatter plot of score-distance vs kernel-distance between primal and mutants ...")


# make scatter plot of score-distance vs kernel-distance between primal and mutants
plt.scatter(kernelDistancePMVals, scoreDistancePMVals)#, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None)
# plt.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None)
plt.title("Distance between Primal and its Mutants", fontsize = 10)
plt.xlabel("Kernel-Distance", fontsize = 9)
plt.ylabel("Score-Distance", fontsize = 9)
plt.tight_layout()
plt.savefig(nameScatterPM)
plt.close()
# nameScatterPM = inputFileName.replace(".pkl", "_Scatter_PM.pdf")
# constructionDict["ScoreDistancePMVals"] = deepcopy(scoreDistancePMVals)
# constructionDict["KernelDistancePMVals"] = deepcopy(kernelDistancePMVals)


for i in range(len(kernelDistancePMVals)):
    print(kernelDistancePMVals[i], "\t", scoreDistancePMVals[i])


# task message
print("\n")
print("*** Making scatter plot of score-distance vs kernel-distance between pairs of mutants ...")


# make scatter plot of score-distance vs kernel-distance between pairs of mutants
plt.scatter(kernelDistanceMMVals, scoreDistanceMMVals)#, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None)
# plt.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None)
plt.title("Distance between Pairs of Mutants", fontsize = 10)
plt.xlabel("Kernel-Distance", fontsize = 9)
plt.ylabel("Score-Distance", fontsize = 9)
plt.tight_layout()
plt.savefig(nameScatterMM)
plt.close()
# nameScatterMM = inputFileName.replace(".pkl", "_Scatter_MM.pdf")
# constructionDict["ScoreDistanceMMVals"] = deepcopy(scoreDistanceMMVals)
# constructionDict["KernelDistanceMMVals"] = deepcopy(kernelDistanceMMVals)


for i in range(len(kernelDistanceMMVals)):
    print(kernelDistanceMMVals[i], "\t", scoreDistanceMMVals[i])


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
