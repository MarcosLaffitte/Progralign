################################################################################
#                                                                              #
#  README - Program: CreatorTool_Analysis.py                                   #
#                                                                              #
#  - Description: produce 30 random non-isomorphic connected Primal graphs,    #
#    whose vertices and edges are randomly labeled with labels from two        #
#    predefined sets, obtaining from them 5 variant classes by deleting or     #
#    adding edges and vertices to them, but preserving the original labels,    #
#    and which are mutations of each other following a "binary tree" fashion.  #
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
"""


# dependencies #################################################################


# installed with conda ---------------------------------------------------------
import networkx as nx
import matplotlib.pyplot as plt


# already in python ------------------------------------------------------------
import os
import math
import random
import pickle as pkl
from copy import deepcopy
from itertools import product, combinations


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# variables ####################################################################


# parameters -------------------------------------------------------------------
sizePrimal = 20
orderPrimal = 10
nonIsoPrimals = 10
levelsOfTree = 4    # including root which is first mutant of primal
totChildren = 2     # children of inner nodes deciding if the tree is n-ary
# class 1: adding one vertex per level of mutation
rateVertexAdd = 1
# class 2: deleting one vertex per level of mutation
rateVertexRem = 1
# class 3: adding two edges per level of mutation
rateEdgesAdd = 2
# class 4: deleting two edges per level of mutation
rateEdgesRem = 2
# class 5: changing 2 two edges per level of mutation
rateEdgesChg = 2


"""
>>> Data_Smaller
- sizePrimal = 20
- orderPrimal = 10
- nonIsoPrimals = 30
- levelsOfTree = 3
- totChildren = 2

>>> Data_Main
- sizePrimal = 18
- orderPrimal = 10
- nonIsoPrimals = 10
- levelsOfTree = 4
- totChildren = 2
"""


# vertex labels
vLabels = ["vA", "vB", "vC", "vD", "vE"]
# edge labels
eLabels = ["eA", "eB", "eC", "eD", "eE"]


# data holders -----------------------------------------------------------------
newDeg = 0
wiener = 0
radius = 0
newNode = 0
diameter = 0
degrees = []
vDegSeq = []
eDegSeq = []
addEdges = []
neighbors = []
tempLevel = []
deleteNodes = []
deleteEdges = []
currentLevel = []
allInvariants = []
nonRepInvariants = []
nodeLabel = dict()
edgeLabel = dict()
connected = False
newGraph = None
newPrimal = None
donePrimal = None
complement = None


# output -----------------------------------------------------------------------
allPrimal = []
variantClass1 = []     # adding vertices
variantCalss2 = []     # deteling vertices
variantClass3 = []     # adding edges
variantClass4 = []     # deteling edges
variantClass5 = []     # changing edges
outputTuplesVC1 = []
outputTuplesVC2 = []
outputTuplesVC3 = []
outputTuplesVC4 = []
outputTuplesVC5 = []


# main #########################################################################


# create primal connected random graphs
while(len(allPrimal) < nonIsoPrimals):
    # reinitialize flags
    connected = False
    isomorphic = False
    # get connected candidate
    while(not connected):
        newPrimal = nx.gnm_random_graph(orderPrimal, sizePrimal)
        connected = nx.is_connected(newPrimal)
    # evaluate isomorphism with saved primals
    for donePrimal in allPrimal:
        isomorphic = nx.is_isomorphic(newPrimal, donePrimal)
        if(isomorphic):
            break
    # save if new primal and label vertices by degree
    if(not isomorphic):
        # reinitialize labels
        nodeLabel = dict()
        edgeLabel = dict()
        # label vertices
        for v in list(newPrimal.nodes()):
            nodeLabel[v] = random.choice(vLabels)
        # label edges
        for (u, v) in list(newPrimal.edges()):
            edgeLabel[(u, v)] = random.choice(eLabels)
        # asign labels
        nx.set_node_attributes(newPrimal, nodeLabel, name = "vPrimalLabel")
        nx.set_edge_attributes(newPrimal, edgeLabel, name = "ePrimalLabel")
        # save
        allPrimal.append(deepcopy(newPrimal))


# check results with some invariants
for eachPrimal in allPrimal:
    # get vertex degree sequence
    vDegSeq = [eachPrimal.degree(v) for v in list(eachPrimal.nodes())]
    vDegSeq.sort()
    # get edge degree sequence
    eDegSeq = [tuple(sorted([eachPrimal.degree(u), eachPrimal.degree(v)])) for (u, v) in list(eachPrimal.edges())]
    eDegSeq.sort()
    # get radius
    radius = nx.radius(eachPrimal)
    # get diameter
    diameter = nx.diameter(eachPrimal)
    # get wiener index
    wiener = nx.wiener_index(eachPrimal)
    # save invariants
    allInvariants.append((vDegSeq, eDegSeq, radius, diameter, wiener))
    if(not (vDegSeq, eDegSeq, radius, diameter, wiener) in nonRepInvariants):
        nonRepInvariants.append((vDegSeq, eDegSeq, radius, diameter, wiener))
# sort for printing
allInvariants.sort()
for eachTuple in allInvariants:
    print(eachTuple)
# print number of unrepeated invariant tuples
print("\n- Distinct Invariant Tuples: ", len(nonRepInvariants))


print("- Class 1: adding vertices ...")
# produce variant class 1: adding vertices
for eachPrimal in allPrimal:
    # reinitialize class
    currentLevel = []
    variantClass1 = []
    # initialize variant class of the given primal
    newGraph = deepcopy(eachPrimal)
    newNode = newGraph.order()
    for k in range(rateVertexAdd):
        degrees = list(set([newGraph.degree(v) for v in list(newGraph.nodes())]))
        degrees.sort()
        newDeg = random.choice(degrees)
        neighbors = list(random.sample(list(newGraph.nodes()), newDeg))
        newGraph.add_node(newNode, vPrimalLabel = random.choice(vLabels))
        for eachNeigh in neighbors:
            newGraph.add_edge(newNode, eachNeigh, ePrimalLabel = random.choice(eLabels))
        newNode = newNode + 1
    variantClass1.append(deepcopy(newGraph))
    currentLevel.append(deepcopy(newGraph))
    # iterate over levels
    for i in range(levelsOfTree-1):
        # reinitialize temp level
        tempLevel = []
        # itertae producing the children
        for eachGraph in currentLevel:
            for j in range(totChildren):
                # reinitialize conditions
                isomorphic = True
                # iterate until not isomorphic mutant is produced
                while(isomorphic):
                    # copy current graph
                    newGraph = deepcopy(eachGraph)
                    # get first new name
                    newNode = newGraph.order()
                    # iterate adding number of required new vertices
                    for k in range(rateVertexAdd):
                        # get list of degrees
                        degrees = list(set([newGraph.degree(v) for v in list(newGraph.nodes())]))
                        degrees.sort()
                        newDeg = random.choice(degrees)
                        # choose neighbors
                        neighbors = list(random.sample(list(newGraph.nodes()), newDeg))
                        # add new vertex with random label
                        newGraph.add_node(newNode, vPrimalLabel = random.choice(vLabels))
                        # add neighbors with random label for new edges
                        for eachNeigh in neighbors:
                            newGraph.add_edge(newNode, eachNeigh, ePrimalLabel = random.choice(eLabels))
                        # name of next new vertex (if any)
                        newNode = newNode + 1
                    # check isomorphism with sibling (or last generated graph in general)
                    for doneGraph in variantClass1:
                        isomorphic = nx.is_isomorphic(newGraph, doneGraph)
                        if(isomorphic):
                            break
                # save new graph
                variantClass1.append(deepcopy(newGraph))
                tempLevel.append(deepcopy(newGraph))
        # update current level
        currentLevel = deepcopy(tempLevel)
    # save variant class corresponding to primal
    outputTuplesVC1.append((deepcopy(eachPrimal), deepcopy(variantClass1)))


print("- Class 2: deleting vertices ...")
# produce variant class 2: deleting vertices
for eachPrimal in allPrimal:
    # reinitialize class
    currentLevel = []
    variantClass2 = []
    # initialize variant class of the given primal
    newGraph = deepcopy(eachPrimal)
    deleteNodes = list(random.sample(list(newGraph.nodes()), rateVertexRem))
    newGraph.remove_nodes_from(deleteNodes)
    variantClass2.append(deepcopy(newGraph))
    currentLevel.append(deepcopy(newGraph))
    # iterate over remaining levels producing the required children of each graph
    for i in range(levelsOfTree-1):
        # reinitialize temp level
        tempLevel = []
        # itertae producing over current level producing the children
        for eachGraph in currentLevel:
            for j in range(totChildren):
                # reinitialize conditions
                isomorphic = True
                # iterate until not isomorphic mutant is produced
                while(isomorphic):
                    # copy current graph
                    newGraph = deepcopy(eachGraph)
                    # get vertices to remove
                    deleteNodes = list(random.sample(list(newGraph.nodes()), rateVertexRem))
                    # obtain actual new graph
                    newGraph.remove_nodes_from(deleteNodes)
                    # check isomorphism
                    for doneGraph in variantClass2:
                        isomorphic = nx.is_isomorphic(newGraph, doneGraph)
                        if(isomorphic):
                            break
                # save new graph
                variantClass2.append(deepcopy(newGraph))
                tempLevel.append(deepcopy(newGraph))
        # update current level
        currentLevel = deepcopy(tempLevel)
    # save variant class corresponding to primal
    outputTuplesVC2.append((deepcopy(eachPrimal), deepcopy(variantClass2)))


print("- Class 3: adding edges ...")
# produce variant class 3: adding edges
for eachPrimal in allPrimal:
    # reinitialize class
    currentLevel = []
    variantClass3 = []
    # initialize variant class of the given primal
    newGraph = deepcopy(eachPrimal)
    complement = nx.complement(newGraph)
    addEdges = list(random.sample(list(complement.edges()), rateEdgesAdd))
    newGraph.add_edges_from(addEdges)
    variantClass3.append(deepcopy(newGraph))
    currentLevel.append(deepcopy(newGraph))
    # iterate over levels
    for i in range(levelsOfTree-1):
        # reinitialize temp level
        tempLevel = []
        # itertae producing the children
        for eachGraph in currentLevel:
            for j in range(totChildren):
                # reinitialize conditions
                isomorphic = True
                # iterate until not isomorphic mutant is produced
                while(isomorphic):
                    # copy current graph
                    newGraph = deepcopy(eachGraph)
                    # get complement
                    complement = nx.complement(newGraph)
                    # get edges to add
                    addEdges = list(random.sample(list(complement.edges()), rateEdgesAdd))
                    # obtain actual new graph
                    newGraph.add_edges_from(addEdges)
                    # check isomorphism
                    for doneGraph in variantClass3:
                        isomorphic = nx.is_isomorphic(newGraph, doneGraph)
                        if(isomorphic):
                            break
                # save new graph
                variantClass3.append(deepcopy(newGraph))
                tempLevel.append(deepcopy(newGraph))
        # update current level
        currentLevel = deepcopy(tempLevel)
    # save variant class corresponding to primal
    outputTuplesVC3.append((deepcopy(eachPrimal), deepcopy(variantClass3)))


print("- Class 4: deleting edges ...")
# produce variant class 4: deleting edges
for eachPrimal in allPrimal:
    # reinitialize class
    currentLevel = []
    variantClass4 = []
    # initialize variant class of the given primal
    newGraph = deepcopy(eachPrimal)
    deleteEdges = list(random.sample(list(newGraph.edges()), rateEdgesRem))
    newGraph.remove_edges_from(deleteEdges)
    variantClass4.append(deepcopy(newGraph))
    currentLevel.append(deepcopy(newGraph))
    # iterate over levels
    for i in range(levelsOfTree-1):
        # reinitialize temp level
        tempLevel = []
        # itertae producing the children
        for eachGraph in currentLevel:
            for j in range(totChildren):
                # reinitialize conditions
                isomorphic = True
                # iterate until not isomorphic mutant is produced
                while(isomorphic):
                    # copy current graph
                    newGraph = deepcopy(eachGraph)
                    # get edges to remove
                    deleteEdges = list(random.sample(list(newGraph.edges()), rateEdgesRem))
                    # obtain actual new graph
                    newGraph.remove_edges_from(deleteEdges)
                    # check isomorphism
                    for doneGraph in variantClass4:
                        isomorphic = nx.is_isomorphic(newGraph, doneGraph)
                        if(isomorphic):
                            break
                # save new graph
                variantClass4.append(deepcopy(newGraph))
                tempLevel.append(deepcopy(newGraph))
        # update current level
        currentLevel = deepcopy(tempLevel)
    # save variant class corresponding to primal
    outputTuplesVC4.append((deepcopy(eachPrimal), deepcopy(variantClass4)))


print("- Class 5: changing edges ...")
# produce variant class 5: changing edges
for eachPrimal in allPrimal:
    # reinitialize class
    currentLevel = []
    variantClass5 = []
    # initialize variant class of the given primal
    newGraph = deepcopy(eachPrimal)
    complement = nx.complement(newGraph)
    deleteEdges = list(random.sample(list(newGraph.edges()), rateEdgesRem))
    addEdges = list(random.sample(list(complement.edges()), rateEdgesAdd))
    newGraph.remove_edges_from(deleteEdges)
    newGraph.add_edges_from(addEdges)
    variantClass5.append(deepcopy(newGraph))
    currentLevel.append(deepcopy(newGraph))
    # iterate over levels
    for i in range(levelsOfTree-1):
        # reinitialize temp level
        tempLevel = []
        # itertae producing the children
        for eachGraph in currentLevel:
            for j in range(totChildren):
                # reinitialize conditions
                isomorphic = True
                # iterate until not isomorphic mutant is produced
                while(isomorphic):
                    # copy current graph
                    newGraph = deepcopy(eachGraph)
                    # get complement
                    complement = nx.complement(newGraph)
                    # get edges to remove and edges to add
                    deleteEdges = list(random.sample(list(newGraph.edges()), rateEdgesRem))
                    addEdges = list(random.sample(list(complement.edges()), rateEdgesAdd))
                    # obtain actual new graph
                    newGraph.remove_edges_from(deleteEdges)
                    newGraph.add_edges_from(addEdges)
                    # check isomorphism
                    for doneGraph in variantClass5:
                        isomorphic = nx.is_isomorphic(newGraph, doneGraph)
                        if(isomorphic):
                            break
                # save new graph
                variantClass5.append(deepcopy(newGraph))
                tempLevel.append(deepcopy(newGraph))
        # update current level
        currentLevel = deepcopy(tempLevel)
    # save variant class corresponding to primal
    outputTuplesVC5.append((deepcopy(eachPrimal), deepcopy(variantClass5)))


# create folders
os.system("mkdir Data")
os.system("mkdir Data/Class_1_AddV")
os.system("mkdir Data/Class_2_RemV")
os.system("mkdir Data/Class_3_AddE")
os.system("mkdir Data/Class_4_RemE")
os.system("mkdir Data/Class_5_ChgE")


# save classes
for i in range(len(allPrimal)):
    # save class 1: adding vertices
    os.system("mkdir Data/Class_1_AddV/AddV_" + str(i+1))
    eachTuple = deepcopy(outputTuplesVC1[i])
    outFile = open("Data/Class_1_AddV/AddV_" + str(i+1) + "/AddV_" + str(i+1) + ".pkl", "wb")
    pkl.dump(eachTuple, outFile)
    outFile.close()
    # save class 2: removing vertices
    os.system("mkdir Data/Class_2_RemV/RemV_" + str(i+1))
    eachTuple = deepcopy(outputTuplesVC2[i])
    outFile = open("Data/Class_2_RemV/RemV_" + str(i+1) + "/RemV_" + str(i+1) + ".pkl", "wb")
    pkl.dump(eachTuple, outFile)
    outFile.close()
    # save class 3: adding edges
    os.system("mkdir Data/Class_3_AddE/AddE_" + str(i+1))
    eachTuple = deepcopy(outputTuplesVC3[i])
    outFile = open("Data/Class_3_AddE/AddE_" + str(i+1) + "/AddE_" + str(i+1) + ".pkl", "wb")
    pkl.dump(eachTuple, outFile)
    outFile.close()
    # save class 4: removing edges
    os.system("mkdir Data/Class_4_RemE/RemE_" + str(i+1))
    eachTuple = deepcopy(outputTuplesVC4[i])
    outFile = open("Data/Class_4_RemE/RemE_" + str(i+1) + "/RemE_" + str(i+1) + ".pkl", "wb")
    pkl.dump(eachTuple, outFile)
    outFile.close()
    # save class 5: changing edges
    os.system("mkdir Data/Class_5_ChgE/ChgE_" + str(i+1))
    eachTuple = deepcopy(outputTuplesVC5[i])
    outFile = open("Data/Class_5_ChgE/ChgE_" + str(i+1) + "/ChgE_" + str(i+1) + ".pkl", "wb")
    pkl.dump(eachTuple, outFile)
    outFile.close()


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
