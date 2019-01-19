# -*- coding: utf-8 -*-

import numpy as np
from opts import *
from cube import *
from queue import LifoQueue as Queue

mu = opts["mu"]
c = opts["c"]

n_a = len(moves)
action_set = {a : i for a, i in zip(list(moves), range(n_a))}

class State(object): 

	def __init__(self, state): 
		self.W = np.zeros(n_a)
		self.N = np.zeros(n_a)
		self.L = np.zeros(n_a)
		self.P, self.val = None,None
		self.state_id = state
		self.state_hash = hash_state(state)
		self.totalN = 0
		self.depth = np.inf
		self.min_depth_node = None

	def Q(self):
		return(self.W - self.L)

	def U(self):
		return(c * self.P * np.sqrt(self.totalN)/(1+self.N))

	def loss_update(self, action):
		a = action_set[action]
		self.L[a] += mu
		
	def getVal(self):
		if not self.val:
			P, val = model.predict(reducted_state(self.state_id).reshape(1, 480))
			self.P, self.val = P[0], val[0,0]
			self.W = self.val*np.ones(n_a)
		return(self.val)
		
	def update(self, action, val):
		a = action_set[action]
		self.W[a] = max(self.W[a], val)
		self.N[a] += 1
		self.L[a] -= mu
		self.totalN += 1
	

class StateNode(object):
	def __init__(self, state, action = None, parent = None): # State state, NodeState parent
		
		self.state = state
		self.parent = parent
		self.ActFromParent = action # {StateNode n :action a}
		self.ActToChildren = dict() # {action a : StateNode n}
		self.depth = 0
		self.is_leaf = True
		
		if parent:
			self.depth = min(self.parent.Depth()+1, self.Depth())
		
		if self.depth < self.Depth():
			self.state.depth = self.depth
			self.state.min_depth_node = self
		

	def update(self, action, val):
		self.state.update(action,val)

	def W(self):
		return(self.state.W)

	def N(self):
		return(self.state.N)
		
	def L(self):
		return(self.state.L)

	def Q(self):
		return(self.state.Q())

	def U(self):
		return(self.state.U())
		
	def Depth(self):
		return(self.state.depth)

	def getStateId(self):
		return(self.state.state_id)
	
	def getStateHash(self):
		return(self.state.state_hash)

	def getVal(self):
		return(self.state.getVal())

	def add_children(self, node, action):
		self.ActToChildren[action] = node
		self.is_leaf = False
		
	def get_child(self,a):
		return(self.ActToChildren[a])
	
	def isLeaf(self):
		return(self.is_leaf)
	
	def get_parent(self):
		return(self.parent, self.ActFromParent)
	

class MCTS(object):
	def __init__(self, initial_state):

		self.root = StateNode(State(initial_state))
		self.visited_states = {self.root.getStateHash():self.root.state}

	def __contains__(self, stateId):
		return(stateId in self.visited_states)
	
	def getRoot(self):
		return(self.root)
		
	def add(self, stateId, parent, action):
		stateHash = hash_state(stateId)
		if stateHash not in self:
			new_state = State(stateId)
			new_node = StateNode(new_state, action, parent)
			self.visited_states[stateHash] = new_state
		else:
			new_node = StateNode(self.visited_states[stateHash], action, parent)
		parent.add_children(new_node, action)
		return(new_node)
	
	def BFS(self,state):
		stateHash = hash_state(state.state_id)
		depths = {k : v.min_depth_node for k,v in self.visited_states.items()}
		change = True
		while change:
			change = False
			for k,node in depths.items():
				if node and node.parent and node.parent.Depth()+1 < node.Depth():
					node.state.depth = node.parent.Depth()+1
					change = False
		return(depths[stateHash].Depth(),depths[stateHash])
		
		
	

	

