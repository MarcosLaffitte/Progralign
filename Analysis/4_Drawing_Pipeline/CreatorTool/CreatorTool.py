################################################################################
#                                                                              #
#  README - Program: CreatorTool.py                                            #
#                                                                              #
#  - Paper: Progressive Multiple Alignment of Graphs                           #
#                                                                              #
#  - Github repository: https://github.com/MarcosLaffitte/Progralign           #
#                                                                              #
#  - Date: 10 January 2024                                                     #
#                                                                              #
#  - Contributor(s) to this script:                                            #
#    * @MarcosLaffitte - Marcos E. Gonzalez Laffitte                           #
#                                                                              #
#  - Description: this program is an example on how the networkx graphs can    #
#    be created and saved in a list inside a pickle file. The output of this   #
#    file represents the input for Progralign_Analysis.                        #
#                                                                              #
#  - Input: no input is required, but the graphs can be modified IN-code.      #
#                                                                              #
#  - Output: list of graphs inside a pickle file.                              #
#                                                                              #
#  - Run with (after activating [pgalign] conda environment):                  #
#                                                                              #
#                         python   CreatorTool.py                              #
#                                                                              #
#  - Expected output:  ToyExample.pkl                                          #
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
> Lenguaje: python 3.10.12
> Anaconda: conda 23.7.3
> Packages installed with anaconda:
***** networkx 2.8.4
> Packages already in python:
***** pickle 4.0
"""


# dependencies #################################################################


# installed with conda ---------------------------------------------------------
import networkx as nx


# already in python ------------------------------------------------------------
import pickle as pkl


# turn off warnings and matplotlib to run in the server ------------------------
import warnings
warnings.filterwarnings("ignore")


# main #########################################################################


# initial message
print("\n")
print(">>> CreatorTool - Progralign Github Repository")


# task message
print("\n")
print("*** creating example ...")


# G1
G1 = nx.Graph()
G1.add_edge(1, 2)
G1.add_edge(1, 3)
G1.add_edge(1, 4)
G1.add_edge(1, 5)


# G2
G2 = nx.Graph()
G2.add_edge(1, 2)
G2.add_edge(2, 3)
G2.add_edge(3, 4)
G2.add_edge(4, 5)


# G3
G3 = nx.Graph()
G3.add_edge(1, 2)
G3.add_edge(2, 3)
G3.add_edge(3, 4)
G3.add_edge(4, 1)


# save graphs
graphsList = [G1, G2, G3]
outFile = open("DrawingPaper.pkl", "wb")
pkl.dump(graphsList, outFile)
outFile.close()


# final message
print("\n")
print(">>> Finished")
print("\n")


# end ##########################################################################
################################################################################
