################################################################################
#                                                                              #
#  README - Program: Progralign_Pairwise.py                                    #
#                                                                              #
#  - Description: produces sets of unlabeld graphs by values of order and      #
#    density in order to do a running-time analysis of the MCS detection,      #
#    equivalent to the alignment of two graphs.                                #
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
from math import modf, sqrt, ceil, floor
from operator import eq, itemgetter
from itertools import product, combinations
from sys import argv, exit, getrecursionlimit, setrecursionlimit


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# input and variables ##########################################################


# user input for functionalities -----------------------------------------------
# (default) "order": maximize number of vertices; or "scheme": maximize sum of scores
requestedScore = "order"
# (default) "expand": recursive expansion; or "trimm": iterative trimming
MCSalgorithm = "expand"
# (default) b.True: carry analysis considering ambiguous edges of every inner node of the guide-tree
ambiguousEdges = True
# (default) b.True: only allow alignment of vertices with same labels
respectVLabels = True
# (default) b.True: only allow alignment of edges with same labels
respectELabels = True


# parameters for random graphs -------------------------------------------------
graphsPerCell = 10
orderValues = [8, 9, 10, 11, 12, 13, 14, 15, 16]
densityPairs = [(25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70)]
vLabels = ["vA", "vB", "vC", "vD", "vE"]
eLabels = ["eA", "eB", "eC", "eD", "eE"]


# data holders -----------------------------------------------------------------
count = 0
minSize = 0
maxSize = 0
finalTime = 0
initialTime = 0
analysisTime = 0
lowerProportion = 0
upperProportion = 0
lowerBoundDensity = 0
upperBoundDensity = 0
candidateSize = 0
tempList = []
allIndices = []
sizeInterval = []
graphsSameOrder = []
indexCombinations = []
indexingTuple = ()
allGraphs = dict()
nodeLabels = dict()
edgeLabels = dict()
graphsByCells = dict()
pairwiseAlignment_expand = dict()
pairwiseAlignment_trimm = dict()
newGraph = None
connected = False
isomorphic = True


# output -----------------------------------------------------------------------
outputFile = None
resultsPKL = "Results.pkl"


# functions - general tasks ####################################################


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
        # apply contraction function
        theAlignment = nx.identified_nodes(theAlignment, theMapToParentG1[v1], theMapToParentG2[v2])
        # remove contraction attribute created by networkx
        if("contraction" in list(theAlignment.nodes[theMapToParentG1[v1]].keys())):
            del theAlignment.nodes[theMapToParentG1[v1]]["contraction"]
        # get map to parent
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
def matchScore(someG1, someG2, someMatch, existenceG1, existenceG2, containedG1, containedG2, score = "order"):
    # local variables
    theScore = 0
    # get requested score
    if(score == "order"):
        # optimize MCS based on number of vertices
        theScore = len(someMatch)
    if(score == "scheme"):
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
        # score scheme for arbitrary labels (symmetrical scheme)
        scoreSameLabel = 1
        scoreDiffLabel = 0.1
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
                    if((testExistence[vP][L1] == "-") and (testExistence[vP][L2] == "-")):
                        # double gap
                        sumColumn = sumColumn + scoreGapGap
                    else:
                        # insertion / deletion
                        sumColumn = sumColumn + scoreLabelGap
            # add to overall score
            theScore = theScore + sumColumn
    # end of function
    return(theScore)


# function: delta required by sspk for "symb" and "nsymb" for vertices ---------
# * can be edited in the future for other applications
def kroneckerDeltaOneV(label1, label2):
    # global variables
    global respectVLabels
    # local variables
    resultSame = 1
    if(respectVLabels):
        resultDiff = 0
    else:
        resultDiff = 1
    # end of function
    if(label1 == label2):
        return(resultSame)
    else:
        return(resultDiff)


# function: delta required by sspk for "symb" and "nsymb" for edges ------------
# * can be edited in the future for other applications
def kroneckerDeltaOneE(label1, label2):
    # global variables
    global respectELabels
    # local variables
    resultSame = 1
    if(respectELabels):
        resultDiff = 0
    else:
        resultDiff = 1
    # end of function
    if(label1 == label2):
        return(resultSame)
    else:
        return(resultDiff)


# function: delta required by sspk for "mix" for vertices ----------------------
# * can be edited in the future for other applications
def kroneckerDeltaTwoV(label1, label2, weight1, weight2):
    # global variables
    global respectVLabels
    # local variables
    resultSame = 1
    if(respectVLabels):
        resultDiff = 0
    else:
        resultDiff = 1
    # end of function
    if((label1 == label2) and (weight1 == weight2)):
        return(resultSame)
    else:
        return(resultDiff)


# function: delta required by sspk for "mix" for edges -------------------------
# * can be edited in the future for other applications
def kroneckerDeltaTwoE(label1, label2, weight1, weight2):
    # global variables
    global respectELabels
    # local variables
    resultSame = 1
    if(respectELabels):
        resultDiff = 0
    else:
        resultDiff = 1
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
                      score = "order",
                      ambiguous1 = [], ambiguous2 = [],
                      vLabels = True,
                      eLabels = True,
                      printProgressMCS = False,
                      ambiguousPairsCheck = False,
                      algorithm = "expand"):
    # local variables
    alignmentScore = 0
    match = []
    possibleMatches = []
    isMatchAlsoSubIso = False
    isMatchAlsoMaxColored = False
    # get MCS
    if(algorithm == "expand"):
        if(graphType == "undirected"):
            possibleMatches, isMatchAlsoSubIso, isMatchAlsoMaxColored = undirRecursiveExpansionMCS(someG1, someG2,
                                                                                                   existenceG1, existenceG2,
                                                                                                   containedG1, containedG2,
                                                                                                   ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                                                                   score = score,
                                                                                                   vLabels = vLabels,
                                                                                                   eLabels = eLabels,
                                                                                                   printProgressMCS = printProgressMCS,
                                                                                                   ambiguousPairsCheck = ambiguousPairsCheck)
        if(graphType == "directed"):
            possibleMatches, isMatchAlsoSubIso, isMatchAlsoMaxColored = dirRecursiveExpansionMCS(someG1, someG2,
                                                                                                 existenceG1, existenceG2,
                                                                                                 containedG1, containedG2,
                                                                                                 ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                                                                 score = score,
                                                                                                 vLabels = vLabels,
                                                                                                 eLabels = eLabels,
                                                                                                 printProgressMCS = printProgressMCS,
                                                                                                 ambiguousPairsCheck = ambiguousPairsCheck)
    if(algorithm == "trimm"):
        possibleMatches = iterativeTrimmingMCS(someG1, someG2, graphType,
                                               existenceG1, existenceG2,
                                               containedG1, containedG2,
                                               ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                               score = score,
                                               vLabels = vLabels,
                                               eLabels = eLabels,
                                               printProgressMCS = printProgressMCS,
                                               ambiguousPairsCheck = ambiguousPairsCheck)
    # check if MCS is empty (possible when considering vertex labels)
    if(len(possibleMatches) > 0):
        match = possibleMatches[0]
    else:
        match = []
    # get final score
    alignmentScore = matchScore(someG1, someG2, match,
                                existenceG1, existenceG2,
                                containedG1, containedG2,
                                score = score)
    # end of function
    return(alignmentScore, match)


# function: check if match is actually unlabeld isomorphism --------------------
def isSubIso(someG1, someG2, someMatch):
    # local variables
    edges1 = []
    edges2 = []
    forMatch = dict()
    invMatch = dict()
    notAB = False
    notBA = False
    # get edges
    edges1 = list(someG1.edges())
    edges2 = list(someG2.edges())
    # check cardinality
    if(not len(edges1) == len(edges2)):
        return(False)
    # get forward and inverse matches
    forMatch = {n1:n2 for (n1, n2) in someMatch}
    invMatch = {n2:n1 for (n1, n2) in someMatch}
    # compare edges
    for (u, v) in edges1:
        notAB = (not (forMatch[u], forMatch[v]) in edges2)
        notBA = (not (forMatch[v], forMatch[u]) in edges2)
        if(notAB and notBA):
            return(False)
    for (u, v) in edges2:
        notAB = (not (invMatch[u], invMatch[v]) in edges1)
        notBA = (not (invMatch[v], invMatch[u]) in edges1)
        if(notAB and notBA):
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


# function: modify labels of cherry if requested by the user -------------------
def filterCherry(someGraph, graphType, vLabels = True, eLabels = True):
    # local variables
    filteredGraph = None
    filteredNodes = []
    filteredEdges = []
    # everything is preserved
    if(vLabels and eLabels):
        return(deepcopy(someGraph))
    # initialize graph
    if(graphType == "undirected"):
        filteredGraph = nx.Graph()
    if(graphType == "directed"):
        filteredGraph = nx.DiGraph()
    # only vertex labels are preserved
    if(vLabels and (not eLabels)):
        filteredNodes = list(someGraph.nodes(data = True))
        filteredGraph.add_nodes_from(filteredNodes)
        filteredEdges = list(someGraph.edges())
        filteredEdges = [(u, v, {"edgeLabel":deepcopy(dict())}) for (u, v) in filteredEdges]
        filteredGraph.add_edges_from(filteredEdges)
        return(deepcopy(filteredGraph))
    # only edge labels are preserved
    if((not vLabels) and eLabels):
        filteredNodes = list(someGraph.nodes())
        filteredNodes = [(v, {"nodeLabel":deepcopy(dict())}) for v in filteredNodes]
        filteredGraph.add_nodes_from(filteredNodes)
        filteredEdges = list(someGraph.edges(data = True))
        filteredGraph.add_edges_from(filteredEdges)
        return(deepcopy(filteredGraph))
    # no labels are preserved
    filteredNodes = list(someGraph.nodes())
    filteredNodes = [(v, {"nodeLabel":deepcopy(dict())}) for v in filteredNodes]
    filteredGraph.add_nodes_from(filteredNodes)
    filteredEdges = list(someGraph.edges())
    filteredEdges = [(u, v, {"edgeLabel":deepcopy(dict())}) for (u, v) in filteredEdges]
    filteredGraph.add_edges_from(filteredEdges)
    # end of function
    return(deepcopy(filteredGraph))


# functions - undirected MCS via recursive expansion ###########################


# function: recursive MATCH for undir MCS search -------------------------------
def undirRecursiveExpansionMCS(someG1, someG2,
                               existenceG1, existenceG2,
                               containedG1, containedG2,
                               someMatch = [], allMatches = [],
                               ambiguous1 = [], ambiguous2 = [],
                               score = "order",
                               vLabels = True,
                               eLabels = True,
                               printProgressMCS = True,
                               ambiguousPairsCheck = False,
                               totOrder = dict()):
    # local variables
    progress = 0
    expOrder = 0
    scoreNewMatch = 0
    scoreOldMatch = 0
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
            # save match
            if(len(someMatch) < expOrder):
                allMatches = [someMatch]
                if(vLabels and eLabels):
                    if(isMaxColoredMatch(someG1, someG2, someMatch)):
                        foundMaxColoredMatch = True
                        return(allMatches, foundSubIso, foundMaxColoredMatch)
            else:
                allMatches = [someMatch]
                if(vLabels and eLabels):
                    if(ambiguousPairsCheck):
                        foundSubIso = isSubIso(someG1, someG2, someMatch)
                        if(foundSubIso):
                            foundMaxColoredMatch = True
                    else:
                        foundSubIso = True
                        foundMaxColoredMatch = True
    else:
        # pick score based on arguments
        scoreNewMatch = matchScore(someG1, someG2, someMatch, existenceG1, existenceG2, containedG1, containedG2, score = score)
        scoreOldMatch = matchScore(someG1, someG2, allMatches[0], existenceG1, existenceG2, containedG1, containedG2, score = score)
        # save to MCS list if gets the same score of alignment
        if(scoreNewMatch == scoreOldMatch):
            allMatchesSet = [set(eachMatch) for eachMatch in allMatches]
            if(not set(someMatch) in allMatchesSet):
                # append match
                if(len(someMatch) < expOrder):
                    allMatches = allMatches + [someMatch]
                    if(vLabels and eLabels):
                        if(isMaxColoredMatch(someG1, someG2, someMatch)):
                            foundMaxColoredMatch = True
                            return(allMatches, foundSubIso, foundMaxColoredMatch)
                else:
                    allMatches = allMatches + [someMatch]
                    if(vLabels and eLabels):
                        if(ambiguousPairsCheck):
                            foundSubIso = isSubIso(someG1, someG2, someMatch)
                            if(foundSubIso):
                                foundMaxColoredMatch = True
                        else:
                            foundSubIso = True
                            foundMaxColoredMatch = True
        # overwrite MCS list if there is inprovement in alignment
        if(scoreNewMatch > scoreOldMatch):
            if(len(someMatch) < expOrder):
                allMatches = [someMatch]
                if(vLabels and eLabels):
                    if(isMaxColoredMatch(someG1, someG2, someMatch)):
                        foundMaxColoredMatch = True
                        return(allMatches, foundSubIso, foundMaxColoredMatch)
            else:
                allMatches = [someMatch]
                if(vLabels and eLabels):
                    if(ambiguousPairsCheck):
                        foundSubIso = isSubIso(someG1, someG2, someMatch)
                        if(foundSubIso):
                            foundMaxColoredMatch = True
                    else:
                        foundSubIso = True
                        foundMaxColoredMatch = True
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
                ansSeFy = undirSemanticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, someG1, someG2,
                                                      vLabels = vLabels, eLabels = eLabels)
                if(ansSeFy):
                    # DFS over feasible pairs
                    newMatch = someMatch + [(n1, n2)]
                    allMatches, foundSubIso, foundMaxColoredMatch = undirRecursiveExpansionMCS(someG1, someG2,
                                                                                               existenceG1, existenceG2,
                                                                                               containedG1, containedG2,
                                                                                               newMatch, allMatches,
                                                                                               ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                                                               score = score,
                                                                                               vLabels = vLabels,
                                                                                               eLabels = eLabels,
                                                                                               ambiguousPairsCheck = ambiguousPairsCheck,
                                                                                               totOrder = totOrder)
                    # stop serach if one (sub)graph isomorphism was found (when preseving all labels)
                    if(vLabels and eLabels and foundSubIso):
                        break
                    # stop search if one maximal colored match was found (when preserving all labels)
                    if(vLabels and eLabels and foundMaxColoredMatch):
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
def undirSemanticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, someG1, someG2,
                                vLabels = True, eLabels = True):
    # local variables
    neigh1 = []
    neigh2 = []
    matchNeigh1 = []
    matchNeigh2 = []
    # compare vertex-labels
    if(vLabels):
        if(not someG1.nodes[n1]["nodeLabel"] == someG2.nodes[n2]["nodeLabel"]):
            return(False)
    # compare edge labels
    if(eLabels):
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
            # only true or ambiguous neighbors at this point (this is the intersection)
            if(forMatch[v] in matchNeigh2):
                if(not someG1[n1][v]["edgeLabel"] == someG2[n2][forMatch[v]]["edgeLabel"]):
                    return(False)
    # end of function
    return(True)


# functions - directed MCS via recursive expansion #############################


# function: recursive MATCH for dir MCS search ---------------------------------
def dirRecursiveExpansionMCS(someG1, someG2,
                             existenceG1, existenceG2,
                             containedG1, containedG2,
                             someMatch = [], allMatches = [],
                             ambiguous1 = [], ambiguous2 = [],
                             score = "order",
                             vLabels = True,
                             eLabels = True,
                             printProgressMCS = True,
                             ambiguousPairsCheck = False,
                             totOrder = dict()):
    # local variables
    expOrder = 0
    progress = 0
    scoreNewMatch = 0
    scoreOldMatch = 0
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
            # save match
            if(len(someMatch) < expOrder):
                allMatches = [someMatch]
                if(vLabels and eLabels):
                    if(isMaxColoredMatch(someG1, someG2, someMatch)):
                        foundMaxColoredMatch = True
                        return(allMatches, foundSubIso, foundMaxColoredMatch)
            else:
                allMatches = [someMatch]
                if(vLabels and eLabels):
                    if(ambiguousPairsCheck):
                        foundSubIso = isSubIso(someG1, someG2, someMatch)
                        if(foundSubIso):
                            foundMaxColoredMatch = True
                    else:
                        foundSubIso = True
                        foundMaxColoredMatch = True
    else:
        # pick score based on arguments
        scoreNewMatch = matchScore(someG1, someG2, someMatch, existenceG1, existenceG2, containedG1, containedG2, score = score)
        scoreOldMatch = matchScore(someG1, someG2, allMatches[0], existenceG1, existenceG2, containedG1, containedG2, score = score)
        # save to MCS list if gets the same score of alignment
        if(scoreNewMatch == scoreOldMatch):
            allMatchesSet = [set(eachMatch) for eachMatch in allMatches]
            if(not set(someMatch) in allMatchesSet):
                # append match
                if(len(someMatch) < expOrder):
                    allMatches = allMatches + [someMatch]
                    if(vLabels and eLabels):
                        if(isMaxColoredMatch(someG1, someG2, someMatch)):
                            foundMaxColoredMatch = True
                            return(allMatches, foundSubIso, foundMaxColoredMatch)
                else:
                    allMatches = allMatches + [someMatch]
                    if(vLabels and eLabels):
                        if(ambiguousPairsCheck):
                            foundSubIso = isSubIso(someG1, someG2, someMatch)
                            if(foundSubIso):
                                foundMaxColoredMatch = True
                        else:
                            foundSubIso = True
                            foundMaxColoredMatch = True
        # overwrite MCS list if there is inprovement in alignment
        if(scoreNewMatch > scoreOldMatch):
            if(len(someMatch) < expOrder):
                allMatches = [someMatch]
                if(vLabels and eLabels):
                    if(isMaxColoredMatch(someG1, someG2, someMatch)):
                        foundMaxColoredMatch = True
                        return(allMatches, foundSubIso, foundMaxColoredMatch)
            else:
                allMatches = [someMatch]
                if(vLabels and eLabels):
                    if(ambiguousPairsCheck):
                        foundSubIso = isSubIso(someG1, someG2, someMatch)
                        if(foundSubIso):
                            foundMaxColoredMatch = True
                    else:
                        foundSubIso = True
                        foundMaxColoredMatch = True
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
                ansSeFy = dirSemanticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, someG1, someG2,
                                                    vLabels = vLabels, eLabels = eLabels)
                if(ansSeFy):
                    # DFS over feasible pairs
                    newMatch = someMatch + [(n1, n2)]
                    allMatches, foundSubIso, foundMaxColoredMatch = dirRecursiveExpansionMCS(someG1, someG2,
                                                                                             existenceG1, existenceG2,
                                                                                             containedG1, containedG2,
                                                                                             newMatch, allMatches,
                                                                                             ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                                                             score = score,
                                                                                             vLabels = vLabels,
                                                                                             eLabels = eLabels,
                                                                                             ambiguousPairsCheck = ambiguousPairsCheck,
                                                                                             totOrder = totOrder)
                    # stop serach is one (sub)graph isomorphism was found (when preseving all labels)
                    if(vLabels and eLabels and foundSubIso):
                        break
                    # stop search if one maximal colored match was found (when preseving all labels)
                    if(vLabels and eLabels and foundMaxColoredMatch):
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
def dirSemanticFeasabilityMCS(n1, n2, currMatch1, currMatch2, forMatch, someG1, someG2,
                              vLabels = True, eLabels = True):
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
    # compare edge labels
    if(eLabels):
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
            # only true or ambiguous neighbors at this point (this is the in-intersection)
            if(forMatch[a] in inMatchNeigh2):
                if(not someG1[a][n1]["edgeLabel"] == someG2[forMatch[a]][n2]["edgeLabel"]):
                    return(False)
        for b in outMatchNeigh1:
            # only true or ambiguous neighbors at this point (this is the out-intersection)
            if(forMatch[b] in outMatchNeigh2):
                if(not someG1[n1][b]["edgeLabel"] == someG2[n2][forMatch[b]]["edgeLabel"]):
                    return(False)
    # end of function
    return(True)


# functions - MCS via iterative trimming #######################################


# function: iterative-trimming MCS-search ----------------------------
def iterativeTrimmingMCS(someG1, someG2, someType,
                         existenceG1, existenceG2,
                         containedG1, containedG2,
                         somePreMatch = [],
                         ambiguous1 = [], ambiguous2 = [],
                         score = "order",
                         vLabels = True,
                         eLabels = True,
                         printProgressMCS = True,
                         ambiguousPairsCheck = False):
    # local variables
    K = 0
    smallerWas = 0
    maxNewScore = 0
    maxRemovable = 0
    scoreNewMatch = 0
    scoreOldMatch = 0
    allMatches = []
    newMatches = []
    tempSubIsos = []
    subgraphNodes = []
    preMatchNodes = []
    removableNodes = []
    twistedMatches = []
    tempNewMatches = []
    removableSubsets = []
    smallerGraphNodes = []
    optimalNewMatches = []
    biggerGraph = None
    smallerGraph = None
    trimmedGraph = None
    answerSubIso = False
    # determine smaller graph
    if(someG1.order() >= someG2.order()):
        biggerGraph = deepcopy(someG1)
        smallerGraph = deepcopy(someG2)
        smallerWas = 2
    else:
        biggerGraph = deepcopy(someG2)
        smallerGraph = deepcopy(someG1)
        smallerWas = 1
    # get smaller graph nodes
    smallerGraphNodes = list(smallerGraph.nodes())
    # get removable nodes (not in pre-match)
    if(len(somePreMatch) > 0):
        preMatchNodes = [(n1, n2)[smallerWas-1] for (n1, n2) in somePreMatch]
        removableNodes = [v for v in smallerGraphNodes if(not v in preMatchNodes)]
        maxRemovable = len(removableNodes)
    else:
        removableNodes = deepcopy(smallerGraphNodes)
        maxRemovable = len(smallerGraphNodes)-1
    # iterate removing combinations of vertices
    for K in range(0, maxRemovable+1):
        # reinitialize variables
        newMatches = []
        # get n-choose-k combinations
        removableSubsets = list(combinations(removableNodes, K))
        # remove each subset
        for eachSubset in removableSubsets:
            # reinitialize flags
            answerSubIso = False
            # remove subset of vertices
            trimmedGraph = deepcopy(smallerGraph)
            trimmedGraph.remove_nodes_from(list(eachSubset))
            # test subgraph isomorphism with custome VF2 (adapted for ambiguous edges)
            if(someType == "undirected"):
                answerSubIso, tempSubIsos = undirPseudoVF2(biggerGraph, trimmedGraph, someMatch = somePreMatch,
                                                           ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                           vLabels = vLabels, eLabels = eLabels,
                                                           ambiguousPairsCheck = ambiguousPairsCheck)
            if(someType == "directed"):
                answerSubIso, tempSubIsos = dirPseudoVF2(biggerGraph, trimmedGraph, someMatch = somePreMatch,
                                                         ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                         vLabels = vLabels, eLabels = eLabels,
                                                         ambiguousPairsCheck = ambiguousPairsCheck)
            # save subisos
            if(answerSubIso):
                newMatches = newMatches + tempSubIsos
        # save or return new matches (if any)
        if(len(newMatches) > 0):
            # untwist new matches if it is the case
            if(smallerWas == 1):
                twistedMatches = deepcopy(newMatches)
                tempNewMatches = []
                for eachMatch in twistedMatches:
                    tempNewMatches.append([(n1, n2) for (n2, n1) in eachMatch])
                newMatches = deepcopy(tempNewMatches)
            # if only optimizing score then stop
            if(score == "order"):
                # print final progress and finish
                if(printProgressMCS):
                    printProgress(100, progressIn = "in case", reportCase = False)
                return(newMatches)
            else:
                # if no match so far then save new matches and continue
                if(len(allMatches) == 0):
                    allMatches = deepcopy(newMatches)
                else:
                    # get optimal new matches
                    maxNewScore = 0
                    optimalNewMatches = []
                    for eachMatch in newMatches:
                        scoreNewMatch = matchScore(someG1, someG2, eachMatch, existenceG1, existenceG2, containedG1, containedG2, score = score)
                        if(scoreNewMatch > maxNewScore):
                            optimalNewMatches = [deepcopy(eachMatch)]
                            maxNewScore = scoreNewMatch
                        else:
                            if(scoreNewMatch == maxNewScore):
                                optimalNewMatches.append(deepcopy(eachMatch))
                    # otherwise check if new matches optimize the current score
                    scoreOldMatch = matchScore(someG1, someG2, allMatches[0], existenceG1, existenceG2, containedG1, containedG2, score = score)
                    if(maxNewScore > scoreOldMatch):
                        allMatches = deepcopy(optimalNewMatches)
                    else:
                        if(maxNewScore == scoreOldMatch):
                            allMatches = allMatches + optimalNewMatches
        # print progress
        if(printProgressMCS):
            printProgress(round((K+1)*100/(maxRemovable+1), 2), progressIn = "in case", reportCase = False)
    # end of function
    return(allMatches)


# functions - undir pseudoVF2 (subgraph isomorphism and ambiguous edges) #######


# function: VF2 for undir subgraph isomorphism search with ambiguous edges -----
def undirPseudoVF2(someG1, someSubgraph,
                   someMatch = [], allMatches = [],
                   ambiguous1 = [], ambiguous2 = [],
                   vLabels = True, eLabels = True,
                   ambiguousPairsCheck = False,
                   totOrder = dict()):
    # local variables
    ansSiFy = False
    ansSeFy = False
    ansTemp = False
    newMatch = []
    vertices = []
    currMatch1 = []
    currMatch2 = []
    candidatePairs = []
    forMatch = dict()
    invMatch = dict()
    # define arbitrary total order if not yet defined
    if(len(list(totOrder.keys())) == 0):
        vertices = list(someSubgraph.nodes())
        totOrder = {v: i for (i, v) in list(enumerate(vertices))}
    # get max number of vertices and test match cover
    if(len(someMatch) == someSubgraph.order()):
        # found subgraph isomorphism
        allMatches = allMatches + [someMatch]
    else:
        # generate auxiliary structures
        currMatch1 = [x for (x, y) in someMatch]
        currMatch2 = [y for (x, y) in someMatch]
        forMatch = {x:y for (x, y) in someMatch}
        invMatch = {y:x for (x, y) in someMatch}
        # get candidate pairs (if any)
        candidatePairs = undirCandidatesISO(someMatch, currMatch1, currMatch2, someG1, someSubgraph, totOrder)
        # evaluate candidate pairs
        for (n1, n2) in candidatePairs:
            # evaluate sintactic feasibility
            ansSiFy = undirSintacticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someSubgraph,
                                                   ambiguousPairsCheck, ambiguous1, ambiguous2)
            if(ansSiFy):
                # evaluate semantic feasibility
                ansSeFy = undirSemanticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, someG1, someSubgraph,
                                                      vLabels = vLabels, eLabels = eLabels)
                if(ansSeFy):
                    # DFS over feasible pairs
                    newMatch = someMatch + [(n1, n2)]
                    ansTemp, allMatches = undirPseudoVF2(someG1, someSubgraph,
                                                         newMatch, allMatches,
                                                         ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                         vLabels = vLabels, eLabels = eLabels,
                                                         ambiguousPairsCheck = ambiguousPairsCheck,
                                                         totOrder = totOrder)
    # end of function
    return(len(allMatches) > 0, allMatches)


# function: get candidate pairs for undir isomorphism search -------------------
def undirCandidatesISO(someMatch, currMatch1, currMatch2, someG1, someG2, theOrder):
    # local variables
    minIndex = 0
    P = []
    validNeigh1 = []
    validNeigh2 = []
    alternative1 = []
    alternative2 = []
    # get candidate pairs extending match [T(s)]
    for (n1, n2) in someMatch:
        validNeigh1 = [x for x in list(someG1.neighbors(n1)) if(not x in currMatch1)]
        validNeigh2 = [y for y in list(someG2.neighbors(n2)) if(not y in currMatch2)]
        if((len(validNeigh1) > 0) and (len(validNeigh2) > 0)):
            P = list(set(P + list(product(validNeigh1, validNeigh2))))
    # alternatively try pairing all unpaired vertices [P^d(s)]
    if(len(P) == 0):
        alternative1 = [x for x in list(someG1.nodes()) if(not x in currMatch1)]
        alternative2 = [y for y in list(someG2.nodes()) if(not y in currMatch2)]
        if((len(alternative1) > 0) and (len(alternative2) > 0)):
            P = list(product(alternative1, alternative2))
    # ignore pairs in P not satisfying the total order
    if(len(P) > 0):
        minIndex = min([theOrder[y] for (x, y) in P])
        P = [(x, y) for (x, y) in P if(theOrder[y] == minIndex)]
    # end of function
    return(P)


# function: evaluate the sintactic feasability of mapping n1 to n2 -------------
def undirSintacticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someG2,
                                 ambiguousCheck, ambiguousG1, ambiguousG2):
    # local variables
    neigh1 = []
    neigh2 = []
    ambNeigh1 = []
    ambNeigh2 = []
    matchNeigh1 = []
    matchNeigh2 = []
    exteriorNeigh1 = []
    exteriorNeigh2 = []
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
    # look ahead 1' and 2': G1 should have enough real neighbors and ambiguous neighbors to cover G2 (the subgraph)
    exteriorNeigh1 = [x for x in neigh1 if(not x in currMatch1)]
    exteriorNeigh2 = [y for y in neigh2 if(not y in currMatch2)]
    if(len(exteriorNeigh2) > len(list(set(exteriorNeigh1 + ambNeigh1)))):
        return(False)
    # end of function
    return(True)


# function: evaluate the semantic feasability of mapping n1 to n2 --------------
def undirSemanticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, someG1, someG2,
                                vLabels = True, eLabels = True):
    # local variables
    neigh1 = []
    neigh2 = []
    matchNeigh1 = []
    matchNeigh2 = []
    # compare vertex-labels
    if(vLabels):
        if(not someG1.nodes[n1]["nodeLabel"] == someG2.nodes[n2]["nodeLabel"]):
            return(False)
    # compare edge labels
    if(eLabels):
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
            # only true or ambiguous neighbors at this point (this is the intersection)
            if(forMatch[v] in matchNeigh2):
                if(not someG1[n1][v]["edgeLabel"] == someG2[n2][forMatch[v]]["edgeLabel"]):
                    return(False)
    # end of function
    return(True)


# functions - dir pseudoVF2 (subgraph isomorphism and ambiguous edges) #########


# function: VF2 for dir subgraph isomorphism search with ambiguous edges -------
def dirPseudoVF2(someG1, someSubgraph,
                 someMatch = [], allMatches = [],
                 ambiguous1 = [], ambiguous2 = [],
                 vLabels = True, eLabels = True,
                 ambiguousPairsCheck = False,
                 totOrder = dict()):
    # local variables
    ansSiFy = False
    ansSeFy = False
    ansTemp = False
    newMatch = []
    vertices = []
    currMatch1 = []
    currMatch2 = []
    candidatePairs = []
    forMatch = dict()
    invMatch = dict()
    # define arbitrary total order if not yet defined
    if(len(list(totOrder.keys())) == 0):
        vertices = list(someSubgraph.nodes())
        totOrder = {v: i for (i, v) in list(enumerate(vertices))}
    # get max number of vertices and test match cover
    if(len(someMatch) == someSubgraph.order()):
        # found subgraph isomorphism
        allMatches = allMatches + [someMatch]
    else:
        # generate auxiliary structures
        currMatch1 = [x for (x, y) in someMatch]
        currMatch2 = [y for (x, y) in someMatch]
        forMatch = {x:y for (x, y) in someMatch}
        invMatch = {y:x for (x, y) in someMatch}
        # get candidate pairs (if any)
        candidatePairs = dirCandidatesISO(someMatch, currMatch1, currMatch2, someG1, someSubgraph, totOrder)
        # evaluate candidate pairs
        for (n1, n2) in candidatePairs:
            # evaluate sintactic feasibility
            ansSiFy = dirSintacticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someSubgraph,
                                                 ambiguousPairsCheck, ambiguous1, ambiguous2)
            if(ansSiFy):
                # evaluate semantic feasibility
                ansSeFy = dirSemanticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, someG1, someSubgraph,
                                                    vLabels = vLabels, eLabels = eLabels)
                if(ansSeFy):
                    # DFS over feasible pairs
                    newMatch = someMatch + [(n1, n2)]
                    ansTemp, allMatches = dirPseudoVF2(someG1, someSubgraph,
                                                       newMatch, allMatches,
                                                       ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                       vLabels = vLabels, eLabels = eLabels,
                                                       ambiguousPairsCheck = ambiguousPairsCheck,
                                                       totOrder = totOrder)
    # end of function
    return(len(allMatches) > 0, allMatches)


# function: get candidate pairs for dir isomorphism search ----------------------
def dirCandidatesISO(someMatch, currMatch1, currMatch2, someG1, someG2, theOrder):
    # local variables
    minIndex = 0
    P = []
    alternative1 = []
    alternative2 = []
    inValidNeigh1 = []
    inValidNeigh2 = []
    outValidNeigh1 = []
    outValidNeigh2 = []
    # get candidate pairs of out-neighbors extending match [T_out(s)]
    for (n1, n2) in someMatch:
        outValidNeigh1 = [b1 for b1 in list(someG1.neighbors(n1)) if(not b1 in currMatch1)]
        outValidNeigh2 = [b2 for b2 in list(someG2.neighbors(n2)) if(not b2 in currMatch2)]
        if((len(outValidNeigh1) > 0) and (len(outValidNeigh2) > 0)):
            P = list(set(P + list(product(outValidNeigh1, outValidNeigh2))))
    # alternatively get candidate pairs of in-neighbors extending match [T_in(s)]
    if(len(P) == 0):
        for (n1, n2) in someMatch:
            inValidNeigh1 = [a1 for a1 in list(someG1.predecessors(n1)) if(not a1 in currMatch1)]
            inValidNeigh2 = [a2 for a2 in list(someG2.predecessors(n2)) if(not a2 in currMatch2)]
            if((len(inValidNeigh1) > 0)  and (len(inValidNeigh2) > 0)):
                P = list(set(P + list(product(inValidNeigh1, inValidNeigh2))))
    # alternatively try pairing all unpaired vertices [P^d(s)]
    if(len(P) == 0):
        alternative1 = [x for x in list(someG1.nodes()) if(not x in currMatch1)]
        alternative2 = [y for y in list(someG2.nodes()) if(not y in currMatch2)]
        if((len(alternative1) > 0)  and (len(alternative2) > 0)):
            P = list(product(alternative1, alternative2))
    # ignore pairs in P not satisfying the total order
    if(len(P) > 0):
        minIndex = min([theOrder[y] for (x, y) in P])
        P = [(x, y) for (x, y) in P if(theOrder[y] == minIndex)]
    # end of function
    return(P)


# function: evaluate the sintactic feasability of mapping n1 to n2 -------------
def dirSintacticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someG2,
                               ambiguousCheck, ambiguousG1, ambiguousG2):
    # local variables
    inNeigh1 = []
    inNeigh2 = []
    outNeigh1 = []
    outNeigh2 = []
    ambNeigh1 = []
    ambNeigh2 = []
    inNewstNeigh1 = []
    inNewstNeigh2 = []
    inMatchNeigh1 = []
    inMatchNeigh2 = []
    outMatchNeigh1 = []
    outMatchNeigh2 = []
    outNewstNeigh1 = []
    outNewstNeigh2 = []
    exteriorNeigh1 = []
    exteriorNeigh2 = []
    # get neighbors of n1 and n2
    inNeigh1 = list(someG1.predecessors(n1))
    inNeigh2 = list(someG2.predecessors(n2))
    outNeigh1 = list(someG1.neighbors(n1))
    outNeigh2 = list(someG2.neighbors(n2))
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
    # look ahead 1' and 2': G1 should have enough real neighbors and ambiguous neighbors to cover G2 (the subgraph)
    inNewstNeigh1 = [a1 for a1 in inNeigh1 if(not a1 in currMatch1)]
    inNewstNeigh2 = [a2 for a2 in inNeigh2 if(not a2 in currMatch2)]
    outNewstNeigh1 = [b1 for b1 in outNeigh1 if(not b1 in currMatch1)]
    outNewstNeigh2 = [b2 for b2 in outNeigh2 if(not b2 in currMatch2)]
    exteriorNeigh1 = list(set(inNewstNeigh1 + outNewstNeigh1))
    exteriorNeigh2 = list(set(inNewstNeigh2 + outNewstNeigh2))
    if(len(exteriorNeigh2) > len(list(set(exteriorNeigh1 + ambNeigh1)))):
        return(False)
    # end of function
    return(True)


# function: evaluate the semantic feasability of mapping n1 to n2 --------------
def dirSemanticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, someG1, someG2,
                              vLabels = True, eLabels = True):
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
    # compare edge labels
    if(eLabels):
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
            # only true or ambiguous neighbors at this point (this is the in-intersection)
            if(forMatch[a] in inMatchNeigh2):
                if(not someG1[a][n1]["edgeLabel"] == someG2[forMatch[a]][n2]["edgeLabel"]):
                    return(False)
        for b in outMatchNeigh1:
            # only true or ambiguous neighbors at this point (this is the out-intersection)
            if(forMatch[b] in outMatchNeigh2):
                if(not someG1[n1][b]["edgeLabel"] == someG2[n2][forMatch[b]]["edgeLabel"]):
                    return(False)
    # end of function
    return(True)


# main #########################################################################


# initial message
print("\n")
print(">>> Progralign_Pairwise - Progralign Github Repository")
print("\n")


# task message
print("* Generating indices for graphs ...")


# generate indices for graphs
allIndices = []
for eachOrder in orderValues:
    for eachDensityPair in densityPairs:
        for eachIndex in range(1, graphsPerCell+1):
            indexingTuple = (eachOrder, eachDensityPair, str(eachIndex))
            allIndices.append(indexingTuple)


# task message
print("\n")
print("* Generating and ordering random graphs ...")


# generate random graphs
count = 0
for (eachOrder, (lowerBoundDensity, upperBoundDensity), eachIndex) in allIndices:
    # reinitialize flags
    isomorphic = True
    # loop until obtaining a new non-isomorphic connected graph
    while(isomorphic):
        # reinitialize flags
        connected = False
        # get connected graph
        while(not connected):
            # get candidate number of edges
            lowerProportion = ((eachOrder*(eachOrder-1)/2)/100)*lowerBoundDensity
            upperProportion = ((eachOrder*(eachOrder-1)/2)/100)*upperBoundDensity
            minSize = ceil(lowerProportion)
            maxSize = floor(upperProportion)
            sizeInterval = list(range(minSize, maxSize+1))
            candidateSize = random.choice(sizeInterval)
            # generate candidate
            newGraph = nx.gnm_random_graph(eachOrder, candidateSize)
            connected = nx.is_connected(newGraph)
        # evaluate isomorphism with saved graphs of same order
        graphsSameOrder = [deepcopy(savedGraph) for savedGraph in list(allGraphs.values()) if(savedGraph.order() == newGraph.order())]
        if(len(graphsSameOrder) == 0):
            isomorphic = False
        else:
            for savedGraph in graphsSameOrder:
                isomorphic = nx.is_isomorphic(newGraph, savedGraph)
                if(isomorphic):
                    break
    # label new greaph
    nodeLabels = dict()
    edgeLabels = dict()
    # label vertices
    for v in list(newGraph.nodes()):
        nodeLabels[v] = random.choice(vLabels)
    # label edges
    for (u, v) in list(newGraph.edges()):
        edgeLabels[(u, v)] = random.choice(eLabels)
    # asign labels
    nx.set_node_attributes(newGraph, nodeLabels, name = "vInitialLabel")
    nx.set_edge_attributes(newGraph, edgeLabels, name = "eInitialLabel")
    # save new graph
    allGraphs[(eachOrder, (lowerBoundDensity, upperBoundDensity), eachIndex)] = deepcopy(newGraph)
    # print progress
    count = count + 1
    printProgress(round(count*100/len(allIndices), 2), reportCase = False)


# task message
print("\n")
print("* Preparing data for pairwise analysis ...")


# prepare data
existence = dict()
for eachIndex in allIndices:
    existence[eachIndex] = dict()
    eachGraph = deepcopy(allGraphs[eachIndex])
    for v in list(eachGraph.nodes()):
        existence[eachIndex][v] = {cellIndex:"." for cellIndex in allIndices if(cellIndex[0] == eachIndex[0] and cellIndex[1] == eachIndex[1])}
        existence[eachIndex][v][eachIndex] = deepcopy(condensedLabel(eachGraph).nodes[v]["nodeLabel"])


# define python's recursion limit based on the order of input graphs
scalationVal = 1.5
requiredLimit = max([eachGraph.order() for eachGraph in list(allGraphs.values())])
currentLimit = getrecursionlimit()
if(currentLimit < (scalationVal * requiredLimit)):
    setrecursionlimit(int(scalationVal * requiredLimit))


# make lists of graphs corresponding to same indexing tuple
for eachOrder in orderValues:
    for eachDensityPair in densityPairs:
        # initialize data holders
        tempList = []
        # save graphs in array
        for eachIndex in range(1, graphsPerCell+1):
            tempList.append((str(eachIndex), deepcopy(allGraphs[(eachOrder, eachDensityPair, str(eachIndex))])))
        # save array in dictionary
        graphsByCells[(eachOrder, eachDensityPair)] = deepcopy(tempList)


# initialize pairwise alignment arrays
pairwiseAlignment_expand = dict()
indexVals = [str(i) for i in range(1, graphsPerCell+1)]
indexCombinations = list(combinations(indexVals, r = 2))
for eachOrder in orderValues:
    for eachDensityPair in densityPairs:
        # save (iG1, iG2, distMCS, runningTime)
        pairwiseAlignment_expand[(eachOrder, eachDensityPair)] = [(a, b, 0, 0) for (a, b) in indexCombinations]


# copy for the other analysis
pairwiseAlignment_trimm = deepcopy(pairwiseAlignment_expand)


# task message
print("\n")
print("* Making pairwise MCS-search with recursive expansion ...")


# pairwise alignment
count = 0
allCels = list(pairwiseAlignment_expand.keys())
totParwise = len(allCels)*len(indexCombinations)
for (eachOrder, eachDensityPair) in allCels:
    # get tuples coding pairwise alignments to make
    tempAlignments = []
    alignmentsToMake = deepcopy(pairwiseAlignment_expand[(eachOrder, eachDensityPair)])
    # do alignments corresponding to cell
    for (indexG1, indexG2, distMCS, runningTime) in alignmentsToMake:
        # indexing tuples
        indexingTuple1 = deepcopy((eachOrder, eachDensityPair, indexG1))
        indexingTuple2 = deepcopy((eachOrder, eachDensityPair, indexG2))
        # get graphs
        G1 = deepcopy(condensedLabel(allGraphs[indexingTuple1]))
        G2 = deepcopy(condensedLabel(allGraphs[indexingTuple2]))
        # start timer
        initialTime = time.time()
        # get score between graphs
        scoreBetweenGraphs, matchBetweenGraphs = pairwiseAlignment(G1, G2,
                                                                   existence[indexingTuple1], existence[indexingTuple2],
                                                                   [indexingTuple1], [indexingTuple2],
                                                                   "undirected",
                                                                   score = requestedScore,
                                                                   vLabels = respectVLabels,
                                                                   eLabels = respectELabels,
                                                                   printProgressMCS = True,
                                                                   algorithm = "expand")
        # stop timer
        finalTime = time.time()
        # get analysis time
        analysisTime = finalTime - initialTime
        # save data
        tempAlignments.append((indexG1, indexG2, scoreBetweenGraphs, analysisTime))
        # print progress
        print("\n")
        print("> ", eachOrder, eachDensityPair, indexG1, indexG2)
        print("*** Analysis took: ", analysisTime, " seconds")
        # save order of the MCS and running time
        print("*** Order of MCS: ", len(matchBetweenGraphs), " vertices")
        print("\n")
        print("------------------------------------------------------------------")
    # save results of cell
    pairwiseAlignment_expand[(eachOrder, eachDensityPair)] = deepcopy(tempAlignments)


# task message
print("\n")
print("* Making pairwise MCS-search with iterative trimming ...")


# pairwise alignment
allCels = list(pairwiseAlignment_trimm.keys())
totParwise = len(allCels)*len(indexCombinations)
for (eachOrder, eachDensityPair) in allCels:
    # get tuples coding pairwise alignments to make
    tempAlignments = []
    alignmentsToMake = deepcopy(pairwiseAlignment_trimm[(eachOrder, eachDensityPair)])
    # do alignments corresponding to cell
    for (indexG1, indexG2, distMCS, runningTime) in alignmentsToMake:
        # indexing tuples
        indexingTuple1 = deepcopy((eachOrder, eachDensityPair, indexG1))
        indexingTuple2 = deepcopy((eachOrder, eachDensityPair, indexG2))
        # get graphs
        G1 = deepcopy(condensedLabel(allGraphs[indexingTuple1]))
        G2 = deepcopy(condensedLabel(allGraphs[indexingTuple2]))
        # start timer
        initialTime = time.time()
        # get score between graphs
        scoreBetweenGraphs, matchBetweenGraphs = pairwiseAlignment(G1, G2,
                                                                   existence[indexingTuple1], existence[indexingTuple2],
                                                                   [indexingTuple1], [indexingTuple2],
                                                                   "undirected",
                                                                   score = requestedScore,
                                                                   vLabels = respectVLabels,
                                                                   eLabels = respectELabels,
                                                                   printProgressMCS = True,
                                                                   algorithm = "trimm")
        # stop timer
        finalTime = time.time()
        # get analysis time
        analysisTime = finalTime - initialTime
        # save data
        tempAlignments.append((indexG1, indexG2, scoreBetweenGraphs, analysisTime))
        # print progress
        print("\n")
        print("> ", eachOrder, eachDensityPair, indexG1, indexG2)
        print("*** Analysis took: ", analysisTime, " seconds")
        # save order of the MCS and running time
        print("*** Order of MCS: ", len(matchBetweenGraphs), " vertices")
        print("\n")
        print("------------------------------------------------------------------")
    # save results of cell
    pairwiseAlignment_trimm[(eachOrder, eachDensityPair)] = deepcopy(tempAlignments)


# task message
print("\n")
print("* Saving data ...")


# save results of all experiments
allResults = (pairwiseAlignment_expand,
              pairwiseAlignment_trimm,
              allIndices,
              allGraphs,
              existence,
              graphsByCells)


# save results
outputFile = open(resultsPKL, "wb")
pkl.dump(allResults, outputFile)
outputFile.close()


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
