import numpy as np
import matplotlib.pyplot as plt


def generate_new_data(filename):
    
    data = np.loadtxt(filename, dtype = "int", delimiter = "\t")
    new_data = {}
    for row in data:
        o1, o2, distance = row[0], row[1], row[2]
        if (o1 not in new_data):
            new_data[o1] = {}
        else:
            new_data[o1][o1] = 0
        new_data[o1][o2] = distance
        if (o2 not in new_data):
            new_data[o2] = {}
        else:
            new_data[o2][o2] = 0
        new_data[o2][o1] = distance
        
    return data, new_data


def choose_distant_objects(a, initial_dist, data_dict, count):
    
    b = 0
    dist = initial_dist
    a_dict = data_dict[a]
    for p, length in a_dict.items():
        if length >= dist:
            b = p
            dist = length
    if count == 4:
        return (a, b, initial_dist)
    else:
        return choose_distant_objects(b, dist, data_dict, count + 1)


def find_xi(a, b, max_dist, new_data, Xi):
    
    for item in new_data:
        ai = 0
        bi = 0
        ai = new_data[item][a]
        bi = new_data[item][b]
        xi = (ai ** 2 + max_dist ** 2 - bi ** 2) / (2 * max_dist)
        if item not in Xi:
            Xi[item] = []
        Xi[item].append(xi)
        

def new_dist(new_data, data, Xi):
    for row in data:
        i = row[0]
        j = row[1]
        dist = new_data[i][j]
        new_dist = (dist ** 2 - (Xi[i][-1] - Xi[j][-1]) ** 2) ** 0.5
        new_data[i][j] = new_dist
        new_data[j][i] = new_dist


def plot(filename2):
    
    words = []
    for row in open(filename2, 'r').readlines():
        word = row.strip()
        words.append(word)   
    for i in range(1, 11):
        plt.scatter(Xi[i][0], Xi[i][1]) 
        plt.annotate(words[i-1], (Xi[i][0], Xi[i][1])) 
    plt.show()

         
filename = 'fastmap-data.txt'
filename2 = 'fastmap-wordlist.txt'
k = 2
data, new_data = generate_new_data(filename)
Xi = {}
while k:
    a = np.random.randint(1,11)
    initial_dist = 0
    count = 0
    a, b, max_dist = choose_distant_objects(a, initial_dist, new_data, count)
    print('a,b: {}, {}'.format(a, b))
    find_xi(a, b, max_dist, new_data, Xi)
    new_dist(new_data, data, Xi)
    k -= 1
print('X: {}'.format(Xi))
plot(filename2)
    
