import numpy as np
import math


def read_data(filename):
    
    fin = open(filename, 'r')
    f = fin.readlines()
    states = []
    towers = []
    observations = []
    
    for i in range(2, 12):
        row = f[i].strip('\n').split(' ')
        for j in range(len(row)):
                element = row[j]
                if element == '1':
                    states.append([i-2, j])

    for i in range(16, 20):
        row = f[i].split(': ')[1].strip('\n').split(' ')
        towers.append([int(row[0]), int(row[1])])

    for i in range(24, 35):
        row = f[i].replace('  ', ' ').strip('\n').split(' ')
        observations.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        
    return states, towers, observations          


def cal_tp(states):
    
    final_transition_probability = np.zeros((len(states), len(states)))  
    transition_probability = {}
    for i in range(10):
        transition_probability[i] = {}
        for j in range(10):
            count = 0
            if [i,j] in states:
                transition_probability[i][j] = {}
                if [i+1,j] in states:
                    count += 1
                if [i-1,j] in states:
                    count += 1
                if [i,j+1] in states:
                    count += 1
                if [i,j-1] in states:
                    count += 1
                
                if [i+1,j] in states:
                    transition_probability[i][j][i+1,j] = 1/count
                if [i-1,j] in states:
                    transition_probability[i][j][i-1,j] = 1/count
                if [i,j+1] in states:
                    transition_probability[i][j][i,j+1] = 1/count
                if [i,j-1] in states:
                    transition_probability[i][j][i,j-1] = 1/count
    
    for i in transition_probability:
        for j in transition_probability[i]:
            for key in transition_probability[i][j]:
                key_index = list(key)
                final_transition_probability[states.index([i,j])][states.index(key_index)] = transition_probability[i][j][key]
    
    return final_transition_probability


def cal_point_ep(distances):
    
    temporary_list = []
    for d in distances:
        if d == 0:
            temporary_list.append(0)
        else:
            temporary_list.append((1.3 - 0.7) / .1 + 1)
        probability = 1
        for i in temporary_list:
            probability = probability * (1 / i)
            
    return probability


def cal_ep(states, observations):
    
    emission_probability = np.zeros((len(observations), len(states)))
    for i in range(len(states)):  
        distances = []
        for j in range(len(towers)):
            euclidean_distance = math.sqrt(pow(states[i][0] - towers[j][0],2) + pow(states[i][1] - towers[j][1],2))
            distances.append(euclidean_distance)
            for k in range(len(observations)):
                counter = 0
                for l in range(len(distances)):
                    a = distances[l] * 0.7
                    b = distances[l] * 1.3
                    if observations[k][l] >= a and observations[k][l] <= b:
                        counter += 1
                    if counter == 4:
                        emission_probability[k][i] = cal_point_ep(distances)
                    
    return emission_probability


def compare_prob(i, j):
    
     if i == 0 or j == 0 or abs(i - j) >= 1e-24:
            
        return i > j


def viterbi(observations, states, ep, tp):
    
    probability = np.zeros((len(observations), len(states)))
    previous_state = np.zeros((len(observations), len(states)))
    previous_state = previous_state.astype(int)
    for i in range(len(states)):
        probability[0][i] = ep[0][i]
    for i in range(1, len(observations)):
        for j in range(len(states)):
            max_probability = -1
            for k in range(len(states)):
                if compare_prob(probability[i - 1][k] * tp[k][j] * ep[i - 1][k], max_probability):
                    max_probability = probability[i - 1][k] * tp[k][j] * ep[i - 1][k]
                    probability[i][j] = max_probability
                    previous_state[i][j] = k

    return probability, previous_state


def find_path(states, probability, previous_state):
    
    path = []
    previous_state_index = np.argmax(probability[-1])
    path.append(states[previous_state_index])
 
    for i in range(len(probability) - 1, 0, -1):
        previous_state_index = previous_state[i][previous_state_index]
        path.append(states[previous_state_index])

    return path[::-1]

    
if __name__ == '__main__':
    states, towers, observations = read_data('hmm-data.txt')
    transition_probability = cal_tp(states)
    emission_probability = cal_ep(states, observations)
    probability, previous_state = viterbi(observations, states, emission_probability, transition_probability)
    path = find_path(states, probability, previous_state)
    print(path)

    