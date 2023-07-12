################################################################################
#                                                                              #
#  README - Program: CreatorTool_Ambiguous_Edges.py                            #
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
print(">>> Creator Tool: Ambiguous Edges - Progralign Github Repository")


# task message
print("\n")
print("*** creating sample graphs ...")


# Base
G = nx.Graph()
for k in range(1, 11+1):
    G.add_node(k, labelV = k)
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(4, 5)
G.add_edge(4, 6)
G.add_edge(6, 7)
G.add_edge(7, 8)
G.add_edge(8, 9)
G.add_edge(9, 10)
G.add_edge(10, 11)
G.add_edge(11, 6)
# nx.draw(G, with_labels = True)
# plt.show()


# Extension
H = nx.Graph()
H.add_node(1, labelV = 1)
H.add_node(2, labelV = 2)
H.add_node(3, labelV = 3)
H.add_node(12, labelV = 12)
H.add_node(6, labelV = 6)
H.add_node(7, labelV = 7)
H.add_node(13, labelV = 13)
H.add_node(14, labelV = 14)
H.add_edge(12, 1)
H.add_edge(1, 2)
H.add_edge(2, 3)
H.add_edge(6, 7)
H.add_edge(7, 14)
H.add_edge(14, 13)
# nx.draw(H, with_labels = True)
# plt.show()


# Cover
F = nx.Graph()
F.add_node(1, labelV = 1)
F.add_node(3, labelV = 3)
F.add_node(4, labelV = 4)
F.add_node(5, labelV = 5)
F.add_node(6, labelV = 6)
F.add_node(7, labelV = 7)
F.add_node(8, labelV = 8)
F.add_node(9, labelV = 9)
F.add_node(11, labelV = 11)
F.add_node(12, labelV = 12)
F.add_node(13, labelV = 13)
F.add_node(14, labelV = 14)
F.add_node(15, labelV = 15)
F.add_node(16, labelV = 16)
F.add_edge(1, 12)
F.add_edge(12, 5)
F.add_edge(5, 4)
F.add_edge(4, 3)
F.add_edge(3, 15)
F.add_edge(4, 6)
F.add_edge(6, 11)
F.add_edge(11, 16)
F.add_edge(6, 7)
F.add_edge(7, 14)
F.add_edge(14, 13)
F.add_edge(13, 5)
F.add_edge(7, 8)
F.add_edge(8, 9)
# nx.draw(F, with_labels = True)
# plt.show()


# save graphs
graphsList = [G, H, F]
outFile = open("test_ambiguous_edges.pkl", "wb")
pkl.dump(graphsList, outFile)
outFile.close()


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
