############################### Info ######################################
"""
Author: Muhammad Umar Riaz
"""

############################### Import ####################################
import argparse
from env import Env

############################### Parser ####################################
parser = argparse.ArgumentParser()
# Architecture parameters
parser.add_argument('--classType', type=str, default='CNN', help='Performance evaluation classifier: RNN, CNN.')
parser.add_argument('--budgetVal', type=float, default=0.25, help='Size of the budget: 0.25, 0.50.')
# Training parameters
parser.add_argument('--agentLoading', action='store_true', help='Load trained model.')
parser.add_argument('--agentLoadingFile', type=str, default='./Source/Agent/Agent_Weights_25_RNN.npy', help='Weights filename.')
parser.add_argument('--trainFlag', action='store_true', help='Training flag.')
parser.add_argument('--trainEps', type=int, default=10000, help='Number of training episodes.')
parser.add_argument('--outFreq', type=int, default=100, help='Agent update frequency.')
parser.add_argument('--testFreq', type=int, default=1000, help='Agent testing frequency.')
parser.add_argument('--testStep', type=int, default=100, help='1 for testing on complete test set, other values to set the testing index step.')
# Input/Output Dirs
parser.add_argument('--dataDir', type=str, default='./Source/InData/', help='Input data directory.')
parser.add_argument('--outDir', type=str, default='./Output/RNN-25/', help='Output directory.')
args = parser.parse_args()

############################### Main #####################################
Env = Env(args)
if args.trainFlag:
    Env.training()
else:
    Env.testing()
