import random
import numpy as np
from collections import defaultdict
import bisect
import collections
import heapq
import operator
import os.path
import random
import math
import functools
from itertools import chain, combinations

f = open("input.txt","r")
out = open("output.txt","w")
input = f.readlines()

lines = [x.strip() for x in input]

grid_size = int(lines[0])
car_count = int(lines[1])
obstacles_count = int(lines[2])

grid1 = np.full((grid_size, grid_size), -1)

for line in lines[3:obstacles_count+3]:
    ind = line.split(',')
    i = int(ind[0])
    j = int(ind[1])
    grid1[j][i] = grid1[j][i] - 100

# print grid
start_pos = lines[obstacles_count + 3:obstacles_count + 3 + car_count]
end_pos = lines[obstacles_count + 3 + car_count:]


orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, -1), (-1, 0), (0, 1)]
turns = LEFT, RIGHT = (+1, -1)


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def add(a, b):
    return tuple(map(operator.add, a, b))


def turn_right(heading):
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    return turn_heading(heading, LEFT)


cc = -1
ans_list = []
for car in start_pos:
    cc = cc + 1
    terminals = end_pos[cc]
    grid = grid1.copy()
    grid[int(terminals[2])][int(terminals[0])] = grid[int(terminals[2])][int(terminals[0])] + 100
    class calculator:
        def __init__(self, grid, terminals, init=(0, 0), reward={}, gamma=0.9, alist=orientations, transitions={}):
            self.terminals = terminals
            self.gamma = gamma
            self.reward = reward
            states = set()
            self.grid = grid
            for x in range(grid_size):
                for y in range(grid_size):
                    if grid[y][x]:
                        states.add((x, y))
                        reward[(x, y)] = grid[y][x]
            self.states = states
            self.alist = alist
            self.transitions = transitions
            for s in states:
                transitions[s] = {}
                for a in alist:
                    transitions[s][a] = self.trans_prob(s, a)

        def actions(self, state):
            return self.alist

        def get_reward(self, state):
            return self.reward[state]

        def trans_prob(self, state, action):
                return [(0.7, self.goto_state(state, action)),
                        (0.1, self.goto_state(state, turn_right(action))),
                        (0.1, self.goto_state(state, turn_right(turn_right(action)))),
                        (0.1, self.goto_state(state, turn_left(action)))]

        def trans(self, state, action):
            return self.transitions[state][action]

        def goto_state(self, state, direction):
            state1 = add(state, direction)
            return state1 if state1 in self.states else state


    cal = calculator(grid, terminals)

    def values(calc, epsilon=0.1):
        new_utility = {s: 0 for s in calc.states}
        rew, tr, gamma = calc.get_reward, calc.trans, calc.gamma

        while True:
            util = new_utility.copy()
            delta = 0
            for s in calc.states:
                if rew(s) == 99:
                    new_utility[s] = 99
                    # print("Continue")
                    continue

                new_utility[s] = rew(s) + gamma * max(sum(p * util[s1] for (p, s1) in tr(s, a))
                                           for a in calc.actions(s))
                delta = max(delta, abs(new_utility[s] - util[s]))
            # print(U1)
            if delta <= epsilon * (1 - gamma) / gamma:
                return util


    def best(calc, utility):
        pi = {}
        for s in calc.states:
            pi[s] = max(calc.actions(s), key=lambda a: exp(a, s, utility, calc))
        # print pi
        return pi


    def exp(a, s, utility, calc):
        return sum(p * utility[s1] for (p, s1) in calc.trans(s, a))

    u = values(cal)
    policy = best(cal, u)

    car = car.split(",")
    l = []
    l = ((int(car[0]), int(car[1])))
    list = []
    for j in range(10):
        pos = l
        cost = 0
        np.random.seed(j)
        swerve = np.random.random_sample(1000000)
        k = 0
        # print cost
        while(cal.get_reward(pos) != 99):
            # print pos
            move = policy[pos]
            if swerve[k] > 0.7:
                if swerve[k] > 0.8:
                    if swerve[k] > 0.9:
                        move = turn_right(turn_right(move))
                        cost = cost + cal.get_reward(pos)
                        pos = cal.goto_state(pos, move)
                    else:
                        move = turn_right(move)
                        cost = cost + cal.get_reward(pos)
                        pos = cal.goto_state(pos, move)
                else:
                    move = turn_left(move)
                    # print move
                    cost = cost + cal.get_reward(pos)
                    pos = cal.goto_state(pos, move)
            else:
                # print move
                cost = cost + cal.get_reward(pos)
                pos = cal.goto_state(pos, move)
            k = k + 1
        cost = cost + 100
        list.append(cost)
    x = sum(list)/float(10)
    x2 = np.floor(x)
    x3 = int(x2)
    out.write("%i" % (x3))
    ans_list.append(x3)
# print ans_list

