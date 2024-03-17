################################################################################
#                                                                              #
#  README - Program: Progralign_Analysis.py                                    #
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
#  - Description: receives a list of arbitrary (possibly labeled) NetworkX     #
#    (di-)graphs (all of the same type, be it undirected or directed) saved in #
#    in a "*.pkl" file, and carries the progressive alignment of these graphs  #
#    by following a guide tree built with WPGMA on graph kernel similarities.  #
#                                                                              #
#  - Input: a python list of graphs inside a pickle file like the one created  #
#    by the CreatorTool in the Progralign repository. These should be either   #
#    of NetworkX's types Graph or DiGraph.                                     #
#                                                                              #
#  - Output: a pickled python dictionary with the results of the alignment.    #
#                                                                              #
#  - Run with (after activating [pgalign] conda environment):                  #
#                                                                              #
#    * default:  python  Progralign_Analysis.py  ToyExample.pkl                #
#    * or:       python  Progralign_Analysis.py  --expand  ToyExample.pkl      #
#                                                                              #
#    By default the program runs the iterative trimming algorithm for MCS      #
#    search, but the --expand option runs instead the recursive expansion      #
#    based on the VF2 algorithm modified for MCS search, please find more      #
#    details in the Paper cited above.                                         #
#                                                                              #
#  - Expected output: example_Results.pkl                                      #
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
***** graphkit-learn 0.2.1
> Packages already in python:
***** pickle 4.0
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


# user input for drawing and functionalities -----------------------------------
# (default) "trimm": iterative trimming; or "expand": recursive expansion
MCSalgorithm = "trimm"
# (default) "order": maximize number of vertices; or "scheme": maximize sum of scores with the scheme in-code
requestedScore = "order"
# (default) "distance": guide tree built on distance; or "similarity": built on the value returned by the kernel
guideTreeMeasure = "distance"
# (default) b.True: carry analysis considering ambiguous edges of every inner node of the guide-tree
ambiguousEdges = True
# (default) b.True: only allow alignment of vertices with same labels
respectVLabels = True
# (default) b.True: only allow alignment of edges with same labels
respectELabels = True


# input ------------------------------------------------------------------------
inputFileName = ""
inputList = []
inputFile = None


# data holders -----------------------------------------------------------------
leafCount = 0
pairScoreG1G2 = 0
uvDistanceSSP = 0
uvSimilaritySSP = 0
theType = ""
dendrogramSSP = ""
kConsensusVals = []
existence = dict()
mapToRoot = dict()
consensusGraphs = dict()
constructionDict = dict()
guideTreeSSP = None
finalAlignment = None
kernelMatrixSSP  = None
clusteringGraphSSP = None


# check user input options -----------------------------------------------------
if(len(argv) in [2, 3]):
    if(len(argv) == 2):
        if(".pkl" in argv[1]):
            remainder = (argv[1].split(".pkl"))[-1]
            if(not remainder == ""):
                errorStr = "\n >> Progralign: Wrong input extension.\n"
                errorStr = errorStr + "- Expected: *.pkl\n"
                errorStr = errorStr + "- Received: *.pkl" + remainder + "\n"
                exit(errorStr)
            else:
                inputFileName = argv[1]
        else:
            exit("\n >> Progralign: Wrong input format.\n")
    if(len(argv) == 3):
        if((argv[1] == "--expand") and (".pkl" in argv[2])):
            remainder = (argv[2].split(".pkl"))[-1]
            if(not remainder == ""):
                errorStr = "\n >> Progralign: Wrong input extension.\n"
                errorStr = errorStr + "- Expected: *.pkl\n"
                errorStr = errorStr + "- Received: *.pkl" + remainder + "\n"
                exit(errorStr)
            else:
                inputFileName = argv[2]
                MCSalgorithm = "expand"
        else:
            exit("\n >> Progralign: Wrong input format.\n")
else:
    exit("\n >> Progralign: Wrong input format.\n")


# output -----------------------------------------------------------------------
outputFile = None
nameResultsPKL = inputFileName.replace(".pkl", "_Results.pkl")


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
def buildAlignment(someG1, someG2, someMatch,
                   existenceG1, existenceG2,
                   containedG1, containedG2):
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
        theLeaves = deepcopy(containedG1 + containedG2)
        forwardMatch = {v1:v2 for (v1, v2) in someMatch}
        for v in classA:
            vP = theMapToParentG1[v]
            testExistence[vP] = dict()
            for leaf in theLeaves:
                if(leaf in containedG1):
                    testExistence[vP][leaf] = deepcopy(existenceG1[v][leaf])
                if(leaf in containedG2):
                    testExistence[vP][leaf] = deepcopy(existenceG2[forwardMatch[v]][leaf])
        for v in classB:
            vP = theMapToParentG1[v]
            testExistence[vP] = dict()
            for leaf in theLeaves:
                if(leaf in containedG1):
                    testExistence[vP][leaf] = deepcopy(existenceG1[v][leaf])
                if(leaf in containedG2):
                    testExistence[vP][leaf] = "-"
        for v in classC:
            vP = theMapToParentG2[v]
            testExistence[vP] = dict()
            for leaf in theLeaves:
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


# function: get consensus graphs for a given construction dictionary -----------
def getConsensusGraphs(someAlignment, someExistence, someLeaves, someConsensusVals):
    # local variables
    count = 0
    allVertices = []
    kthVertices = []
    badVertices = []
    consensusGraph = None
    proportion = dict()
    allConsensusGraphs = dict()
    # get vertices of alignment
    allVertices = list(someAlignment.nodes())
    # get presence of vertices of alignment in input graphs
    for v in allVertices:
        # reinitialize variables
        count = 0
        # loop getting proportions
        for eachLeaf in someLeaves:
            if(not someExistence[v][eachLeaf] == "-"):
                count = count + 1
        # save proportion
        proportion[v] = count
    # get all k-consensus graphs
    for (eachK, totGraphs) in someConsensusVals:
        # reinitialize variables
        kthVertices = [v for v in allVertices if(proportion[v] >= eachK)]
        # induce graph
        consensusGraph = deepcopy(someAlignment)
        badVertices = [v for v in allVertices if(not v in kthVertices)]
        consensusGraph.remove_nodes_from(badVertices)
        allConsensusGraphs[(eachK, totGraphs)] = deepcopy(consensusGraph)
    # end of function
    return(deepcopy(allConsensusGraphs))


# function: get number of gaps in columns of alignment -------------------------
def getGaps(someAlignment, someExistence, someLeaves):
    # local variables
    totGaps = 0
    tempGaps = 0
    vertices = []
    partialGaps = []
    # get vertices of alignmet
    vertices = list(someAlignment.nodes())
    # loop obtaninng total number of gaps
    for v in vertices:
        # get number of gaps corresponding to each vertex
        tempGaps = len([eachLeaf for eachLeaf in someLeaves if(someExistence[v][eachLeaf] == "-")])
        # increase gaps
        totGaps = totGaps + tempGaps
    # end of function
    return(totGaps)


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
        totOrder = {v: i for (i, v) in list(enumerate(list(someG2.nodes()), start = 1))}
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
    # get candidates preserving order (if no previous match just take everything)
    if(len(someMatch) > 0):
        maxMatchedIndex = max([totOrder[n2] for (n1, n2) in someMatch])
    # get candidate pairs
    valid1 = [x for x in list(someG1.nodes()) if(not x in currMatch1)]
    valid2 = [y for y in list(someG2.nodes()) if((not y in currMatch2) and (totOrder[y] > maxMatchedIndex))]
    P = list(product(valid1, valid2))
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
        totOrder = {v: i for (i, v) in list(enumerate(list(someG2.nodes()), start = 1))}
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
    # get candidates preserving order (if no previous match just take everything)
    if(len(someMatch) > 0):
        maxMatchedIndex = max([totOrder[n2] for (n1, n2) in someMatch])
    # alternatively try pairing all unpaired vertices
    valid1 = [x for x in list(someG1.nodes()) if(not x in currMatch1)]
    valid2 = [y for y in list(someG2.nodes()) if((not y in currMatch2) and (totOrder[y] > maxMatchedIndex))]
    P = list(product(valid1, valid2))
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


# function: iterative-trimming MCS-search --------------------------------------
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
    subsetCount = 0
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
    untwistedPreMatch = []
    removableSubsets = []
    smallerGraphNodes = []
    optimalNewMatches = []
    biggerGraph = None
    smallerGraph = None
    trimmedGraph = None
    answerSubIso = False
    # change this if exhaustive enumeration is needed
    justOneMatch = True
    # determine smaller graph
    if(someG1.order() >= someG2.order()):
        biggerGraph = deepcopy(someG1)
        smallerGraph = deepcopy(someG2)
        smallerWas = 2
        untwistedPreMatch = deepcopy(somePreMatch)
    else:
        biggerGraph = deepcopy(someG2)
        smallerGraph = deepcopy(someG1)
        smallerWas = 1
        untwistedPreMatch = [(n2, n1) for (n1, n2) in somePreMatch]
    # get smaller graph nodes
    smallerGraphNodes = list(smallerGraph.nodes())
    # get removable nodes (not in pre-match)
    if(len(somePreMatch) > 0):
        preMatchNodes = [b for (a, b) in untwistedPreMatch]
        removableNodes = [v for v in smallerGraphNodes if(not v in preMatchNodes)]
        maxRemovable = len(removableNodes)
    else:
        removableNodes = deepcopy(smallerGraphNodes)
        maxRemovable = len(smallerGraphNodes)-1
    # iterate removing combinations of vertices
    subsetCount = 0
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
                answerSubIso, tempSubIsos = undirPseudoVF2(biggerGraph, trimmedGraph, someMatch = untwistedPreMatch,
                                                           ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                           vLabels = vLabels, eLabels = eLabels,
                                                           ambiguousPairsCheck = ambiguousPairsCheck,
                                                           oneMatch = justOneMatch)
            if(someType == "directed"):
                answerSubIso, tempSubIsos = dirPseudoVF2(biggerGraph, trimmedGraph, someMatch = untwistedPreMatch,
                                                         ambiguous1 = ambiguous1, ambiguous2 = ambiguous2,
                                                         vLabels = vLabels, eLabels = eLabels,
                                                         ambiguousPairsCheck = ambiguousPairsCheck,
                                                         oneMatch = justOneMatch)
            # print progress
            if(printProgressMCS):
                subsetCount = subsetCount + 1
                printProgress(round(subsetCount*100/(2**maxRemovable), 2), progressIn = "in case", reportCase = False)
            # save subisos
            if(answerSubIso):
                newMatches = newMatches + tempSubIsos
                # early finish when optimizing order and just one match is needed
                if(justOneMatch and (score == "order")):
                    break
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
    # end of function
    return(allMatches)


# functions - undir pseudoVF2 (subgraph isomorphism and ambiguous edges) #######


# function: VF2 for undir subgraph isomorphism search with ambiguous edges -----
def undirPseudoVF2(someG1, someSubgraph,
                   someMatch = [], allMatches = [],
                   ambiguous1 = [], ambiguous2 = [],
                   vLabels = True, eLabels = True,
                   ambiguousPairsCheck = False,
                   oneMatch = True,
                   totOrder = dict()):
    # local variables
    ansSiFy = False
    ansSeFy = False
    ansTemp = False
    T1 = []
    T2 = []
    newMatch = []
    vertices = []
    currMatch1 = []
    currMatch2 = []
    candidatePairs = []
    forMatch = dict()
    invMatch = dict()
    # define arbitrary total order if not yet defined
    # for consistency with "expand" enumerate starts in 1
    if(len(list(totOrder.keys())) == 0):
        vertices = list(someSubgraph.nodes())
        totOrder = {v: i for (i, v) in list(enumerate(vertices, start = 1))}
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
        candidatePairs, T1, T2 = undirCandidatesISO(someMatch, currMatch1, currMatch2, someG1, someSubgraph, totOrder)
        # evaluate candidate pairs
        for (n1, n2) in candidatePairs:
            # evaluate sintactic feasibility
            ansSiFy = undirSintacticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someSubgraph,
                                                   T1, T2, ambiguousPairsCheck, ambiguous1, ambiguous2)
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
                                                         oneMatch = oneMatch,
                                                         totOrder = totOrder)
                    # finish when finding one match
                    if(oneMatch and ansTemp):
                        break
    # end of function
    return(len(allMatches) > 0, allMatches)


# function: get candidate pairs for undir isomorphism search -------------------
def undirCandidatesISO(someMatch, currMatch1, currMatch2, someG1, someG2, theOrder):
    # local variables
    minIndex = 0
    P = []
    neighT1 = []
    neighT2 = []
    validNeigh1 = []
    validNeigh2 = []
    alternative1 = []
    alternative2 = []
    # get candidate pairs extending match [T(s)]
    for (n1, n2) in someMatch:
        validNeigh1 = [x for x in list(someG1.neighbors(n1)) if(not x in currMatch1)]
        validNeigh2 = [y for y in list(someG2.neighbors(n2)) if(not y in currMatch2)]
        neighT1 = list(set(neighT1 + validNeigh1))
        neighT2 = list(set(neighT2 + validNeigh2))
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
    return(P, neighT1, neighT2)


# function: evaluate the sintactic feasability of mapping n1 to n2 -------------
def undirSintacticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someG2,
                                 someT1, someT2, ambiguousCheck, ambiguousG1, ambiguousG2):
    # local variables
    neigh1 = []
    neigh2 = []
    ambNeigh1 = []
    ambNeigh2 = []
    matchNeigh1 = []
    matchNeigh2 = []
    candiNeigh1 = []
    candiNeigh2 = []
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
    # look ahead 1': G1 should provide enough neighbors and ambiguous neighbors in T to cover G2 (the subgraph)
    candiNeigh1 = [x for x in neigh1 if(x in someT1)]
    candiNeigh2 = [y for y in neigh2 if(y in someT2)]
    if(len(candiNeigh2) > len(list(set(candiNeigh1 + ambNeigh1)))):
        return(False)
    # look ahead 2': G1 should have enough real neighbors and ambiguous neighbors to cover G2 (the subgraph)
    exteriorNeigh1 = [x for x in neigh1 if((not x in someT1) and (not x in currMatch1))]
    exteriorNeigh2 = [y for y in neigh2 if((not y in someT2) and (not y in currMatch2))]
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
                 oneMatch = True,
                 totOrder = dict()):
    # local variables
    ansSiFy = False
    ansSeFy = False
    ansTemp = False
    inT1 = []
    inT2 = []
    outT1 = []
    outT2 = []
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
        totOrder = {v: i for (i, v) in list(enumerate(vertices, start = 1))}
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
        candidatePairs, inT1, inT2, outT1, outT2 = dirCandidatesISO(someMatch, currMatch1, currMatch2, someG1, someSubgraph, totOrder)
        # evaluate candidate pairs
        for (n1, n2) in candidatePairs:
            # evaluate sintactic feasibility
            ansSiFy = dirSintacticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someSubgraph,
                                                 inT1, inT2, outT1, outT2, ambiguousPairsCheck, ambiguous1, ambiguous2)
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
                                                       oneMatch = oneMatch,
                                                       totOrder = totOrder)
                    # finish when finding one match
                    if(oneMatch and ansTemp):
                        break
    # end of function
    return(len(allMatches) > 0, allMatches)


# function: get candidate pairs for dir isomorphism search ----------------------
def dirCandidatesISO(someMatch, currMatch1, currMatch2, someG1, someG2, theOrder):
    # local variables
    minIndex = 0
    P = []
    inNeighT1 = []
    inNeighT2 = []
    outNeighT1 = []
    outNeighT2 = []
    alternative1 = []
    alternative2 = []
    inValidNeigh1 = []
    inValidNeigh2 = []
    outValidNeigh1 = []
    outValidNeigh2 = []
    # get candidate pairs of out-neighbors extending match [T_out(s)]
    for (n1, n2) in someMatch:
        # out neighborhood required to build pairs
        outValidNeigh1 = [b1 for b1 in list(someG1.neighbors(n1)) if(not b1 in currMatch1)]
        outValidNeigh2 = [b2 for b2 in list(someG2.neighbors(n2)) if(not b2 in currMatch2)]
        outNeighT1 = list(set(outNeighT1 + outValidNeigh1))
        outNeighT2 = list(set(outNeighT2 + outValidNeigh2))
        # in neighborhood for sintactic feasibility
        inValidNeigh1 = [a1 for a1 in list(someG1.predecessors(n1)) if(not a1 in currMatch1)]
        inValidNeigh2 = [a2 for a2 in list(someG2.predecessors(n2)) if(not a2 in currMatch2)]
        inNeighT1 = list(set(inNeighT1 + inValidNeigh1))
        inNeighT2 = list(set(inNeighT2 + inValidNeigh2))
        # build pairs
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
    return(P, inNeighT1, inNeighT2, outNeighT1, outNeighT2)


# function: evaluate the sintactic feasability of mapping n1 to n2 -------------
def dirSintacticFeasabilityISO(n1, n2, currMatch1, currMatch2, forMatch, invMatch, someG1, someG2,
                               inT1, inT2, outT1, outT2, ambiguousCheck, ambiguousG1, ambiguousG2):
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
    inCandiNeigh1 = []
    inCandiNeigh2 = []
    outCandiNeigh1 = []
    outCandiNeigh2 = []
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
    # look ahead 1' : G1 should povide enough real neighbors and ambiguous neighbors in T to cover G2 (the subgraph)
    inCandiNeigh1 = [a1 for a1 in inNeigh1 if(a1 in inT1)]
    inCandiNeigh2 = [a2 for a2 in inNeigh2 if(a2 in inT2)]
    if(len(inCandiNeigh2) > len(list(set(inCandiNeigh1 + ambNeigh1)))):
        return(False)
    outCandiNeigh1 = [b1 for b1 in outNeigh1 if(b1 in outT1)]
    outCandiNeigh2 = [b2 for b2 in outNeigh2 if(b2 in outT2)]
    if(len(outCandiNeigh2) > len(list(set(outCandiNeigh1 + ambNeigh1)))):
        return(False)
    # look ahead 2': G1 should have enough real neighbors and ambiguous neighbors to cover G2 (the subgraph)
    inNewstNeigh1 = [a1 for a1 in inNeigh1 if((not a1 in inT1) and (not a1 in currMatch1))]
    inNewstNeigh2 = [a2 for a2 in inNeigh2 if((not a2 in inT2) and (not a2 in currMatch2))]
    if(len(inNewstNeigh2) > len(list(set(inNewstNeigh1 + ambNeigh1)))):
        return(False)
    outNewstNeigh1 = [b1 for b1 in outNeigh1 if((not b1 in outT1) and (not b1 in currMatch1))]
    outNewstNeigh2 = [b2 for b2 in outNeigh2 if((not b2 in outT2) and (not b2 in currMatch2))]
    if(len(outNewstNeigh2) > len(list(set(outNewstNeigh1 + ambNeigh1)))):
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


# functions - progressive graph alignment ######################################


# function: complete PGA based on a given guide tree ---------------------------
def PGA(someType, someGT, someBaseDict, someDendro, someContained, someScore,
        holdV = True, holdE = True, holdAmbEdges = True, algorithm = "expand"):
    # local variables
    counter = 0
    heightGT = 0
    finalTime = 0
    finalScore = 0
    initialTime = 0
    alignmentTime = 0
    match = []
    toRemove = []
    cherries = []
    allLeaves = []
    leafNeighbors = []
    possibleMatches = []
    resutls = ()
    theAlignmentResults = ()
    pathRoot = dict()
    allScores = dict()
    mapToRoot = dict()
    existence = dict()
    alignments = dict()
    mapToParent = dict()
    depthLeaves = dict()
    allAlignments = dict()
    ambiguousPairs = dict()
    allLeavesGraphs = dict()
    isMatchAlsoSubIso = False
    isMatchAlsoMaxColored = False
    G1 = None
    G2 = None
    guideTree = None
    graphLeaf = None
    finalAlignment = None
    # prepare data for alignment
    guideTree = deepcopy(someGT)
    # get leaves as current alignments
    alignments = {h:deepcopy(condensedLabel(someBaseDict[h])) for h in list(guideTree.nodes()) if(guideTree.degree(h) == 1)}
    allLeaves = [h for h in list(guideTree.nodes()) if(guideTree.degree(h) == 1)]
    allLeavesGraphs = deepcopy(alignments)   # saves only leaves at the end
    allAlignments = deepcopy(alignments)     # saves leaves and all alignments at the end
    # get leaves values for existence dictionary and their depths
    existence = dict()
    for eachLeaf in list(alignments.keys()):
        allScores[eachLeaf] = 0
        mapToRoot[eachLeaf] = dict()
        existence[eachLeaf] = dict()
        graphLeaf = deepcopy(alignments[eachLeaf])
        ambiguousPairs[eachLeaf] = []
        for v in list(graphLeaf.nodes()):
            mapToRoot[eachLeaf][v] = ""
            existence[eachLeaf][v] = {str(i):"." for i in indices}
            existence[eachLeaf][v][eachLeaf] = deepcopy(graphLeaf.nodes[v]["nodeLabel"])
    # get paths from leafs to root and depth of leaves
    pathRoot = {h:nx.shortest_path(guideTree, source = h, target = someDendro) for h in list(alignments.keys())}
    # get height of guide tree
    depthLeaves = {h:int(nx.shortest_path_length(guideTree, source = h, target = someDendro)) for h in list(alignments.keys())}
    heightGT = max([int(nx.shortest_path_length(guideTree, source = h, target = someDendro)) for h in list(alignments.keys())])
    # start timer
    initialTime = time.time()
    # progressive graph alignment based on guide-tree pruning
    mapToParent = dict()
    counter = 0
    while(len(list(alignments.keys())) > 1):
        # FIRST get vertices adjacent to exactly two leaves
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
            if(someType == "undirected"):
                # get MCS
                G1 = deepcopy(alignments[x])
                G2 = deepcopy(alignments[y])
                if(algorithm == "expand"):
                    possibleMatches, isMatchAlsoSubIso, isMatchAlsoMaxColored = undirRecursiveExpansionMCS(G1, G2,
                                                                                                           existence[x], existence[y],
                                                                                                           someContained[x], someContained[y],
                                                                                                           ambiguous1 = ambiguousPairs[x], ambiguous2 = ambiguousPairs[y],
                                                                                                           score = someScore,
                                                                                                           vLabels = holdV,
                                                                                                           eLabels = holdE,
                                                                                                           ambiguousPairsCheck = holdAmbEdges)
                if(algorithm == "trimm"):
                    possibleMatches = iterativeTrimmingMCS(G1, G2, someType,
                                                           existence[x], existence[y],
                                                           someContained[x], someContained[y],
                                                           ambiguous1 = ambiguousPairs[x], ambiguous2 = ambiguousPairs[y],
                                                           score = someScore,
                                                           vLabels = holdV,
                                                           eLabels = holdE,
                                                           ambiguousPairsCheck = holdAmbEdges)
                # check if MCS is empty (possible when considering vertex labels)
                if(len(possibleMatches) > 0):
                    match = possibleMatches[0]
                else:
                    match = []
                # build alignment graph and get final score
                resutls = buildAlignment(G1, G2, match,
                                         existence[x], existence[y],
                                         someContained[x], someContained[y])
                allScores[cherry] = matchScore(G1, G2, match,
                                               existence[x], existence[y],
                                               someContained[x], someContained[y],
                                               score = someScore)
                # store alignment graph and its attributes
                alignments[cherry] = deepcopy(filterCherry(resutls[0], someType, vLabels = holdV, eLabels = holdE))
                allAlignments[cherry] = deepcopy(alignments[cherry])
                existence[cherry] = deepcopy(resutls[1])
                mapToParent[x] = deepcopy(resutls[2])
                mapToParent[y] = deepcopy(resutls[3])
                ambiguousPairs[cherry] = getAmbiguousPairs(alignments[cherry], existence[cherry], someContained[cherry])
            # alignment for directed graphs
            if(someType == "directed"):
                # get MCS
                G1 = deepcopy(alignments[x])
                G2 = deepcopy(alignments[y])
                if(algorithm == "expand"):
                    possibleMatches, isMatchAlsoSubIso, isMatchAlsoMaxColored = dirRecursiveExpansionMCS(G1, G2,
                                                                                                         existence[x], existence[y],
                                                                                                         someContained[x], someContained[y],
                                                                                                         ambiguous1 = ambiguousPairs[x], ambiguous2 = ambiguousPairs[y],
                                                                                                         score = someScore,
                                                                                                         vLabels = holdV,
                                                                                                         eLabels = holdE,
                                                                                                         ambiguousPairsCheck = holdAmbEdges)
                if(algorithm == "trimm"):
                    possibleMatches = iterativeTrimmingMCS(G1, G2, someType,
                                                           existence[x], existence[y],
                                                           someContained[x], someContained[y],
                                                           ambiguous1 = ambiguousPairs[x], ambiguous2 = ambiguousPairs[y],
                                                           score = someScore,
                                                           vLabels = holdV,
                                                           eLabels = holdE,
                                                           ambiguousPairsCheck = holdAmbEdges)
                # check if MCS is empty (possible when considering vertex labels)
                if(len(possibleMatches) > 0):
                    match = possibleMatches[0]
                else:
                    match = []
                # build alignment graph and get final score
                resutls = buildAlignment(G1, G2, match,
                                         existence[x], existence[y],
                                         someContained[x], someContained[y])
                allScores[cherry] = matchScore(G1, G2, match,
                                               existence[x], existence[y],
                                               someContained[x], someContained[y],
                                               score = someScore)
                # store alignment graph and its attributes
                alignments[cherry] = deepcopy(filterCherry(resutls[0], someType, vLabels = holdV, eLabels = holdE))
                allAlignments[cherry] = deepcopy(alignments[cherry])
                existence[cherry] = deepcopy(resutls[1])
                mapToParent[x] = deepcopy(resutls[2])
                mapToParent[y] = deepcopy(resutls[3])
                ambiguousPairs[cherry] = getAmbiguousPairs(alignments[cherry], existence[cherry], someContained[cherry])
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
    # retrieve final alignment and other data
    finalAlignment = alignments[someDendro]
    finalAlignment = deepcopy(extendedLabel(finalAlignment))
    finalScore = allScores[someDendro]
    # make tuple of results
    theAlignmentResults = (deepcopy(finalAlignment),
                           deepcopy(allAlignments),
                           deepcopy(existence),
                           (someScore, finalScore, holdV, holdE),
                           deepcopy(ambiguousPairs),
                           deepcopy(mapToRoot),
                           deepcopy(allLeaves),
                           deepcopy(allLeavesGraphs),
                           alignmentTime)
    # end of function
    return(theAlignmentResults)


# main #########################################################################


# initial message
print("\n")
print(">>> Progralign Analysis - Progralign Github Repository")


# task message
print("\n")
print("* Retrieving input file ...")


# retrieve input file
inputFile = open(inputFileName, "rb")
inputList = pkl.load(inputFile)
inputFile.close()


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
testG = nx.Graph()
testD = nx.DiGraph()
for eachGraph in inputList:
    if(not type(eachGraph) in [type(testG), type(testD)]):
        exit("\n >> Progralign: the file " + inputFileName + " contains an object which is not of NetworkX's type Graph or DiGraph.\n")
if(not len(list(set([type(eachGraph) for eachGraph in inputList]))) == 1):
    exit("\n >> Progralign: the file " + inputFileName + " contains both directed and undirected graphs. These must be of the same type.\n")
if(len(inputList) == 1):
    exit("\n >> Progralign: the file " + inputFileName + " contains only one (di-)graph. Alignment won't be computed.\n")
for eachGraph in inputList:
    if(eachGraph.order() == 0):
        exit("\n >> Progralign: the file " + inputFileName + " contains a (di-)graph with no vertices. Alignment won't be computed.\n")
if(nx.is_directed(inputList[0])):
    theType = "directed"
else:
    theType = "undirected"


# define python's recursion limit based on the order of input graphs
scalationVal = 1.5
requiredLimit = max([eachGraph.order() for eachGraph in inputList])
currentLimit = getrecursionlimit()
if(currentLimit < (scalationVal * requiredLimit)):
    setrecursionlimit(int(scalationVal * requiredLimit))


# task message
print("* Data is consistent. Indexing graphs ...")


# generate indices for the input graphs
# *** the keys are saved as str just in case
# *** we want to change them in the future
indices = list(range(len(inputList)))
for i in indices:
    constructionDict[str(i)] = deepcopy(inputList[i])
constructionDict["Indices"] = deepcopy(indices)


# task message
print("* Evaluating Structural-Shortest-Path graph-kernel ...")


# evaluate graph-kernels
inputListCondensed = [deepcopy(condensedLabelToStr(eachGraph)) for eachGraph in inputList]
nodeKernels = {"symb": kroneckerDeltaOneV, "nsymb": kroneckerDeltaOneV, "mix": kroneckerDeltaTwoV}
edgeKernels = {"symb": kroneckerDeltaOneE, "nsymb": kroneckerDeltaOneE, "mix": kroneckerDeltaTwoE}
kernelMatrixSSP = sspk.structuralspkernel(inputListCondensed, parallel = None, verbose = True,
                                          node_label = "nodeLabelStr", edge_label = "edgeLabelStr",
                                          node_kernels = nodeKernels, edge_kernels = edgeKernels)[0]
constructionDict["SimilarityMatrix"] = deepcopy(kernelMatrixSSP)


# task message
print("\n")
print("* Retrieving pairwise similarities and distances between input graphs ...")


# build similarity graph
clusteringGraphSSP = nx.Graph()
for (i, j) in list(combinations(indices, r = 2)):
    # build similarity based on kernel-distance
    if(guideTreeMeasure == "distance"):
        selfScoreG1 = kernelMatrixSSP[i][i]
        selfScoreG2 = kernelMatrixSSP[j][j]
        pairScoreG1G2 = kernelMatrixSSP[i][j]
        uvDistanceSSP = sqrt(selfScoreG1 + selfScoreG2 - 2*pairScoreG1G2)
        clusteringGraphSSP.add_edge(str(i), str(j), measure = uvDistanceSSP)
    # build similarity based on kernel-similarity
    if(guideTreeMeasure == "similarity"):
        uvSimilaritySSP = kernelMatrixSSP[i][j]
        clusteringGraphSSP.add_edge(str(i), str(j), measure = uvSimilaritySSP)


# task message
print("* Building guide tree ...")


# iterative construction of dendrogram
dendrogramSSP, guideTreeSSP, containedGraphsSSP = getGuideTree(clusteringGraphSSP, guideTreeMeasure)
constructionDict["GTMeasure"] = guideTreeMeasure
constructionDict["Dendrogram"] = dendrogramSSP
constructionDict["GuideTree"] = deepcopy(guideTreeSSP)
constructionDict["Contained"] = deepcopy(containedGraphsSSP)



# task message
print("\n")
print("* Progressive graph alignment - algorithm: " + MCSalgorithm + " - With Ambiguous Edges")
# progressive graph alignment
alignmentResults = PGA(theType, guideTreeSSP, constructionDict, dendrogramSSP,
                       containedGraphsSSP, requestedScore,
                       holdV = respectVLabels, holdE = respectELabels,
                       holdAmbEdges = ambiguousEdges,
                       algorithm = MCSalgorithm)
# unpack results
constructionDict["Alignment"] = deepcopy(alignmentResults[0])
constructionDict["allAlignments"] = deepcopy(alignmentResults[1])
constructionDict["Existence"] = deepcopy(alignmentResults[2])
constructionDict["ScoreInfo"] = deepcopy(alignmentResults[3])
constructionDict["AmbiguousEdges"] = deepcopy(alignmentResults[4])
constructionDict["MapToRoot"] = deepcopy(alignmentResults[5])
constructionDict["AllLeaves"] = deepcopy(alignmentResults[6])
constructionDict["AllLeavesGraphs"] = deepcopy(alignmentResults[7])
constructionDict["Time"] = deepcopy(alignmentResults[8])
constructionDict["GapsAndAEs"] = (getGaps(constructionDict["Alignment"],
                                          constructionDict["Existence"][constructionDict["Dendrogram"]],
                                          constructionDict["AllLeaves"]),
                                  len(constructionDict["AmbiguousEdges"][constructionDict["Dendrogram"]]))


# task message
print("\n")
print("* Obtaining consensus graphs for proportions of input graphs ...")


# get k-consensus keys
kConsensusVals = [(k, len(indices)) for k in range(1, len(indices)+1)]


# get consensus graphs
consensusGraphs = getConsensusGraphs(constructionDict["Alignment"],
                                     constructionDict["Existence"][constructionDict["Dendrogram"]],
                                     constructionDict["AllLeaves"],
                                     kConsensusVals)
constructionDict["ConsensusGraphs"] = deepcopy(consensusGraphs)
constructionDict["kConsensusVals"] = deepcopy(kConsensusVals)


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


# end ##########################################################################
################################################################################
