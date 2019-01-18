import argparse
from network import *
import numpy as np

ap = argparse.ArgumentParser()

ap.add_argument("-lname", "--modelloadname", type=str, default = "weightstot58new",
                help="model name")
ap.add_argument("-mu", "--mu", type=float, default = 1000,
                help="mu hyperparameter")
ap.add_argument("-T", "--max_exploration", type=int, default = 25000,
                help="max_exploration")
ap.add_argument("-c", "--c", type=float, default = 40,
                help="c hyperparameter")
ap.add_argument("-s", "--save_path", type=str, default = "C:/Users/Cyril/Documents/MVA/RL/rubiks_cube/models/",
                help="save_path")
ap.add_argument("-mp", "--model_path", type=str, default = "C:/Users/Cyril/Documents/MVA/RL/rubiks_cube/models/",
                help="model_path")
opts = vars(ap.parse_args())


