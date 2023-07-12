################################################################################
#                                                                              #
#  README - Program: CreatorTool_Cycles.py                                     #
#                                                                              #
#  - Paper: [-]                                                                #
#                                                                              #
#  - Github repository: https://github.com/MarcosLaffitte/Progralign           #
#                                                                              #
#  - Date: 01 August 2023                                                      #
#                                                                              #
#  - Contributor(s) to this script:                                            #
#    * @MarcosLaffitte - Marcos E. Gonzalez Laffitte                           #
#                                                                              #
#  - Description: vertion of CreatorTool that produces 3 graphs with cycles.   #                
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
import random
import pickle as pkl
from copy import deepcopy
from itertools import product, combinations


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
warnings.filterwarnings("ignore")


# main #########################################################################


# initial message
print("\n")
print(">>> Creator Tool: Cycles - Progralign Github Repository")


# task message
print("\n")
print("*** creating sample graphs ...")


# C3
I = nx.Graph()
I.add_node("1", labelV1 = "A", labelV2 = 0.1)
I.add_node("2", labelV1 = "A", labelV2 = 0.1)
I.add_node("3", labelV1 = "A", labelV2 = 0.1)
I.add_edge("1", "2")
I.add_edge("2", "3")
I.add_edge("3", "1")
nx.draw(I, with_labels = True)
plt.show()


# C4
J = nx.Graph()
J.add_node("1", labelV1 = "B", labelV2 = 0.3)
J.add_node("2", labelV1 = "B", labelV2 = 0.3)
J.add_node("3", labelV1 = "B", labelV2 = 0.3)
J.add_node("4", labelV1 = "B", labelV2 = 0.3)
J.add_edge("1", "2")
J.add_edge("2", "3")
J.add_edge("3", "4")
J.add_edge("4", "1")
nx.draw(J, with_labels = True)
plt.show()


# C3-C4
F = nx.Graph()
F.add_node("1", labelV1 = "A", labelV2 = 0.1)
F.add_node("2", labelV1 = "A", labelV2 = 0.1)
F.add_node("3", labelV1 = "A", labelV2 = 0.1)
F.add_edge("1", "2")
F.add_edge("2", "3")
F.add_edge("3", "1")
F.add_node("4", labelV1 = "I1", labelV2 = 0.2)
F.add_edge("1", "4")
F.add_edge("4", "5")
F.add_node("5", labelV1 = "B", labelV2 = 0.3)
F.add_node("6", labelV1 = "B", labelV2 = 0.3)
F.add_node("7", labelV1 = "B", labelV2 = 0.3)
F.add_node("8", labelV1 = "B", labelV2 = 0.3)
F.add_edge("5", "6")
F.add_edge("6", "7")
F.add_edge("7", "8")
F.add_edge("8", "5")
nx.draw(F, with_labels = True)
plt.show()


# C4-C5
G = nx.Graph()
G.add_node("1", labelV1 = "B", labelV2 = 0.3)
G.add_node("2", labelV1 = "B", labelV2 = 0.3)
G.add_node("3", labelV1 = "B", labelV2 = 0.3)
G.add_node("4", labelV1 = "B", labelV2 = 0.3)
G.add_edge("1", "2")
G.add_edge("2", "3")
G.add_edge("3", "4")
G.add_edge("4", "1")
G.add_node("5", labelV1 = "I2", labelV2 = 0.4)
G.add_edge("2", "5")
G.add_edge("5", "6")
G.add_node("6", labelV1 = "C", labelV2 = 0.5)
G.add_node("7", labelV1 = "C", labelV2 = 0.5)
G.add_node("8", labelV1 = "C", labelV2 = 0.5)
G.add_node("9", labelV1 = "C", labelV2 = 0.5)
G.add_node("10", labelV1 = "C", labelV2 = 0.5)
G.add_edge("6", "7")
G.add_edge("7", "8")
G.add_edge("8", "9")
G.add_edge("9", "10")
G.add_edge("10", "6")
nx.draw(G, with_labels = True)
plt.show()


# C5-C6
H = nx.Graph()
H.add_node("1", labelV1 = "C", labelV2 = 0.5)
H.add_node("2", labelV1 = "C", labelV2 = 0.5)
H.add_node("3", labelV1 = "C", labelV2 = 0.5)
H.add_node("4", labelV1 = "C", labelV2 = 0.5)
H.add_node("5", labelV1 = "C", labelV2 = 0.5)
H.add_edge("1", "2")
H.add_edge("2", "3")
H.add_edge("3", "4")
H.add_edge("4", "5")
H.add_edge("5", "1")
H.add_node("6", labelV1 = "I3", labelV2 = 0.6)
H.add_edge("1", "6")
H.add_edge("6", "7")
H.add_node("7", labelV1 = "D", labelV2 = 0.7)
H.add_node("8", labelV1 = "D", labelV2 = 0.7)
H.add_node("9", labelV1 = "D", labelV2 = 0.7)
H.add_node("10", labelV1 = "D", labelV2 = 0.7)
H.add_node("11", labelV1 = "D", labelV2 = 0.7)
H.add_node("12", labelV1 = "D", labelV2 = 0.7)
H.add_edge("7", "8")
H.add_edge("8", "9")
H.add_edge("9", "10")
H.add_edge("10", "11")
H.add_edge("11", "12")
H.add_edge("12", "7")
nx.draw(H, with_labels = True)
plt.show()


# save graphs
graphsList = [F, G, H, I, J]
outFile = open("test_cycles.pkl", "wb")
pkl.dump(graphsList, outFile)
outFile.close()


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
