import numpy as np
import random

class Cube:

    def __init__(self, state_beg=np.arange(48)):
        self.labels = np.copy(state_beg)

    def move(self, move, in_place = False):

        """
        :param move: string de 2 char représentant le move
        :return: nouveau cube, reward
        """

        if move[0] == "u":
            cycle = u_cycle
        elif move[0] == "d":
            cycle = d_cycle
        elif move[0] == "f":
            cycle = f_cycle
        elif move[0] == "b":
            cycle = b_cycle
        elif move[0] == "l":
            cycle = l_cycle
        elif move[0] == "r":
            cycle = r_cycle

        new_cube = Cube()
        new_cube.labels = np.copy(self.labels)
        if move[1] == "f":
            tmp = np.copy(self.labels[cycle[3]])
            for l in range(3, 0, -1):
                new_cube.labels[cycle[l]] = np.copy(self.labels[cycle[l-1]])
            new_cube.labels[cycle[0]] = tmp
        elif move[1] == "b":
            tmp = np.copy(self.labels[cycle[0]])
            for l in range(3):
                new_cube.labels[cycle[l]] = np.copy(self.labels[cycle[l+1]])
            new_cube.labels[cycle[3]] = tmp
        if in_place:
            self.labels = np.copy(new_cube.labels)
        return new_cube, new_cube.issolve()

    def issolve(self):

        """
        :return: True ssi le cube est dans l'état de départ
        """
        return np.sum(self.labels == np.array(range(48)))==48

    def shuffle(self, n_moves = 10):
        self.labels = np.arange(48)
        for _ in range(n_moves):
            self.move(np.random.choice(moves), True)
        return self

    def reducted_state(self):

        """
        :return: état réduit du cube à donner au réseau de neurones : 20*24 variables 0 ou 1
        """

        res = np.zeros((20,24))
        for i in range(8):
            res[i, matching_corner[self.labels[used_stickers][i]]] = 1
        for i in range(8, 20):
            res[i, matching_edge[self.labels[used_stickers][i]]] = 1
        return res.flatten()
    

import xxhash
def hash_state(state):
    return(state.tostring())

def reducted_state(state):

        """
        :return: état réduit du cube à donner au réseau de neurones : 20*24 variables 0 ou 1
        """

        res = np.zeros((20,24))
        for i in range(8):
            res[i, matching_corner[state[used_stickers][i]]] = 1
        for i in range(8, 20):
            res[i, matching_edge[state[used_stickers][i]]] = 1
        return res.flatten()
    

def inverse(move):
    if move[1]=='f':
        return move['0']+'b'
    else:
        return move['0'] + 'f'

moves = ["uf", "df", "rf", "lf", "ff", "bf", "ub", "db", "rb", "lb", "fb", "bb"]
used_stickers = [0, 1, 2, 3, 40, 41, 42, 43, 4, 5, 6, 7, 20, 24, 22, 26, 44, 45, 46, 47]
corners = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 28, 29, 30, 31, 36, 37, 38, 39, 40, 41, 42, 43]
edges = [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 44, 45, 46, 47]



matching_edge = dict()
matching_corner = dict()

for i, c in enumerate(corners):
    matching_corner[c] = i

for i, e in enumerate(edges):
    matching_edge[e] = i

u_cycle = np.array([
       [ 3,  7, 11, 15, 19],
       [ 2,  6, 10, 14, 18],
       [ 1,  5,  9, 13, 17],
       [ 0,  4,  8, 12, 16]
])

d_cycle = np.array([[40, 44, 28, 32, 36],
       [41, 45, 29, 33, 37],
       [42, 46, 30, 34, 38],
       [43, 47, 31, 35, 39]])


r_cycle = np.array([[ 9, 13, 16, 24, 36],
       [17, 25,  2,  5,  1],
       [37, 33, 30, 22, 10],
       [29, 21, 41, 45, 42]])


l_cycle = np.array([[11, 15,  0,  7,  3],
       [19, 27, 28, 20,  8],
       [39, 35, 43, 47, 40],
       [31, 23, 18, 26, 38]])

f_cycle = np.array([[ 8, 12,  0,  4,  1],
       [16, 24,  9, 21, 29],
       [36, 32, 41, 44, 40],
       [28, 20, 39, 27, 19]])


b_cycle = np.array([[10, 14,  2,  6,  3],
       [18, 26, 11, 23, 31],
       [38, 34, 43, 46, 42],
       [30, 22, 37, 25, 17]])

if __name__ == '__main__':
    cube1 = Cube()
    cube2, r = cube1.move('rf')
    print(r)
    cube3, r = cube2.move('uf')
    print(r)
    cube4, r = cube3.move('ub')
    print(r)
    cube5, r = cube4.move('rb')
    print(r)
    print(cube1.labels)
    print(cube2.labels)
    print(cube3.labels)
    print(cube4.labels)
    print(cube5.labels)
    print(cube4.labels == cube2.labels)
