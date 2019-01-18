import numpy as np
from treelib import Tree

from tree_utils import State, StateNode, MCTS
from network import *
from cube import *
from opts import opts

import time
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare model
model_load_name = opts["model_path"] + "weights/" + opts["modelloadname"] + ".h5"
model.load_weights(model_load_name)
model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)

N_SIMULATIONS = 50
T = opts["max_exploration"]
action_set = list(moves)

solutions = []
histo = []
solve_time = []
distance = []

n_solves = []

n_shuffles = [15]
for n_shuffle in n_shuffles:
	n_solve = 0
	for n in range(N_SIMULATIONS):
		t = time.time()
		i = 1
		cube = Cube()
		initial_state = cube.shuffle(n_shuffle).labels
		
		tree = MCTS(initial_state)
		solved, solved_state = False, None
		
		current_node = tree.getRoot()
		while i < T:
			if current_node.isLeaf() == False:
				A = current_node.Q()+current_node.U()
				a = action_set[np.argmax(A)]
				current_node.state.loss_update(a)
				current_node = current_node.get_child(a)
				
			else:
				# Extend
				cube = Cube(current_node.getStateId())
				for a in action_set:
					new_cube, solved = cube.move(a)
					next_state = new_cube.labels
					new_node = tree.add(next_state, current_node, a)
					if solved:
						solver = new_node
						#print("#%d Solved in %d iterations"%(n,i))
						break
				
				# Backpropagation
				val = current_node.getVal()
				
				A = current_node.Q()+current_node.U()
				a = action_set[np.argmax(A)]
				current_node.state.loss_update(a)
				
				current_node.update(a, val)
				parent, a = current_node.get_parent()
				while parent:
					parent.update(a, val)
					current_node = parent
					parent, a = current_node.get_parent()
					
				# New start
				current_node = tree.getRoot()
				i+=1
			
			if solved:
				break
		
		if solved:
			solution = []
			_, min_solver = tree.BFS(solver.state)
			parent, a = min_solver.get_parent()
			while parent:
				solution = [a] + solution
				parent, a = parent.get_parent()
			n_solve+=1
			histo.append(i)
			solve_time.append(time.time()-t)
			distance.append(len(solution))
	
		else:
#			print("#%d Not Solved :'("%n)
			solution = None
		solutions.append([initial_state,solution])
	
	n_solves.append(n_solve)
	print("Shuffle : %d || Solved : %d/50 || Mean Iteration : %.1f || Max Iteration : %d || Total Time : %.3f s" % (n_shuffle,n_solve, np.mean(histo), max(histo), sum(solve_time)))
	
	sns.distplot(histo)
	plt.title("Number of iterations")
	plt.show()
	
	sns.distplot(solve_time)
	plt.title("Solve Time")
	plt.show()
	
	plt.scatter(distance, solve_time) 
	plt.xlabel("lenghth of the BFS solution") 
	plt.ylabel("solve_time") 
	plt.show()
	
	plt.scatter(histo, solve_time) 
	plt.xlabel("iterations") 
	plt.ylabel("solve_time") 
	plt.show()


