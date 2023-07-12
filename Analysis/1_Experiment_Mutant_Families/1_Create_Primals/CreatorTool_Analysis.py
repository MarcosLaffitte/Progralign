################################################################################
#                                                                              #
#  README - Program: CreatorTool_Analysis.py                                   #
#                                                                              #
#  - Description: produce non-isomorphic Primals and well-behaved mutants for  #
#    them, by vertex-addition and edge-deletion in a binary-tree fashion.      #
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
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match


# already in python ------------------------------------------------------------
import os
import math
import random
import pickle as pkl
from operator import eq
from copy import deepcopy
from itertools import product, combinations


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")


# variables ####################################################################


# parameters -------------------------------------------------------------------
sizePrimal = 40
orderPrimal = 16
nonIsoPrimals = 50
levelsOfTree = 3    # including root which is first mutant of primal
totChildren = 2     # children of inner nodes deciding if the tree is n-ary
# rate of vertex-additions and edge-deletion
rateVertexAdd = [1, 2]
rateVertexRem = [1, 2]


"""
>>> Parameters for Mutants
- sizePrimal = 40  # 33.33% density
- orderPrimal = 16
- nonIsoPrimals = 50
- levelsOfTree = 3
- totChildren = 2
- rateVertexAdd = [1, 2]
- rateVertexRem = [1, 2]
"""


# vertex labels
vLabels = ["vA", "vB", "vC", "vD", "vE"]
# edge labels
eLabels = ["eA", "eB", "eC", "eD", "eE"]


# data holders -----------------------------------------------------------------
newDeg = 0
radius = 0
newNode = 0
diameter = 0
degrees = []
vDegSeq = []
eDegSeq = []
neighbors = []
tempLevel = []
deleteNodes = []
currentLevel = []
allInvariants = []
nonRepInvariants = []
nodeLabel = dict()
edgeLabel = dict()
connected = False
newGraph = None
newPrimal = None
donePrimal = None


# output -----------------------------------------------------------------------
allPrimal = []
variantClass1 = []     # adding vertices and removing edges
outputTuplesVC1 = []


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


# define node match and edge match to be used in MCS search
nodeMatch = generic_node_match("vPrimalLabel", "*", eq)
edgeMatch = generic_edge_match("ePrimalLabel", "*", eq)


# produce mutants in a binary-tree fashion
print("- Generating mutants ...")
for eachPrimal in allPrimal:
    # initialize containers
    currentLevel = []
    variantClass1 = []    
    # reinitialize flag
    connected = False
    # get connected candidate
    while(not connected):
        # get new graph
        newGraph = deepcopy(eachPrimal)
        # get nodes of new graph
        initialNodes = list(newGraph.nodes())
        # get name of first new node
        newNode = newGraph.order()
        # add new nodes
        for k in range(random.choice(rateVertexAdd)):
            degrees = list(set([newGraph.degree(v) for v in list(newGraph.nodes())]))
            degrees.sort()
            newDeg = random.choice(degrees)
            neighbors = list(random.sample(list(newGraph.nodes()), newDeg))
            newGraph.add_node(newNode, vPrimalLabel = random.choice(vLabels))
            for eachNeigh in neighbors:
                newGraph.add_edge(newNode, eachNeigh, ePrimalLabel = random.choice(eLabels))
            newNode = newNode + 1
        # remove nodes randomly from initial nodes
        deleteNodes = list(random.sample(initialNodes, random.choice(rateVertexRem)))
        newGraph.remove_nodes_from(deleteNodes)
        # check connectedness
        connected = nx.is_connected(newGraph)
    # rename vertices (from 0 to n-1) to avoid colision with future graphs
    newGraph = deepcopy(nx.convert_node_labels_to_integers(newGraph))        
    # save new graph
    variantClass1.append(deepcopy(newGraph))
    currentLevel.append(deepcopy(newGraph))
    # iterate over levels
    for i in range(levelsOfTree-1):
        # reinitialize temp level
        tempLevel = []
        # itertae producing the children
        for eachGraph in currentLevel:
            for j in range(totChildren):
                # reinitialize flag
                connected = False
                # get connected candidate
                while(not connected):
                    # copy current graph
                    newGraph = deepcopy(eachGraph)
                    # get nodes of new graph
                    initialNodes = list(newGraph.nodes())
                    # get first new name
                    newNode = newGraph.order()
                    # iterate adding number of required new vertices
                    for k in range(random.choice(rateVertexAdd)):
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
                    # reqmove nodes randomly from initial nodes
                    deleteNodes = list(random.sample(initialNodes, random.choice(rateVertexRem)))
                    newGraph.remove_nodes_from(deleteNodes)
                    # check connectedness
                    connected = nx.is_connected(newGraph)
                # rename vertices (from 0 to n-1) to avoid colision with future graphs
                newGraph = deepcopy(nx.convert_node_labels_to_integers(newGraph))
                # save new graph
                bla = len([(u, v) for (u, v) in list(newGraph.edges()) if(u == v)])
                if(bla > 0):
                    print("NOOOOOOOOOOOOO")
                variantClass1.append(deepcopy(newGraph))
                tempLevel.append(deepcopy(newGraph))
        # update current level
        currentLevel = deepcopy(tempLevel)
    # save variant class corresponding to primal
    outputTuplesVC1.append((deepcopy(eachPrimal), deepcopy(variantClass1)))


# create folders
os.system("mkdir Data")


# save classes
for i in range(len(allPrimal)):
    # save class 1: adding vertices
    os.system("mkdir Data/Mutants_" + str(i+1))
    eachTuple = deepcopy(outputTuplesVC1[i])
    outFile = open("Data/Mutants_" + str(i+1) + "/Mutants_" + str(i+1) + ".pkl", "wb")
    pkl.dump(eachTuple, outFile)
    outFile.close()


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
