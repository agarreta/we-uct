import numpy as np
import math
import re
import random
import networkx as nx
import matplotlib.pyplot as plt


def multinomial(n, k, num):
    return math.factorial(n)/(math.factorial(k)**num)

def moves_wordgame():
    mo = [(a + b, b + a) for a in alphabet for b in alphabet]
    return mo


def moves_hanoi():
    pass

def generate_graph(word, nx_graph=False):
    #moves = [(a + b, b + a) for a in alphabet for b in alphabet]
    #moves = [(a + b, b + a) for i,a in enumerate(alphabet)
    #         for j, b in enumerate(alphabet) if i > j]
    moves = moves_wordgame()

    states = set({word})
    edges = set({})
    words_to_check = set({word})
    if nx_graph:
        G = nx.Graph()
        G.add_node(word)
    else:
        G = None

    def process(word):
        # words_to_check.remove(word)
        for move in moves:
            new_word = re.sub(move[0], move[1], word)

            if new_word != word:
                edges.update({(word, move, new_word)})
                if nx_graph:
                    G.add_edge(word, new_word)
                    G.add_edge(new_word, word)

            if new_word not in states:
                states.update({new_word})
                if nx_graph:
                    G.add_node(new_word)
                words_to_check.update({new_word})
                # generate_states(new_word, states)

    while len(words_to_check) > 0:
        process(words_to_check.pop())
        if len(states) % 1000 == 0:
            print(f'{len(states)} states created')
    return states, edges, G





init_word = 'aabbccdd' # 2500, 64%
init_word = 'aaaaabbbbbccccc'  # 620.000, 63%
init_word = 'aaaabbbbcccc'  # 30,000, 61%
init_word = 'aaaaaaaabbbbbbbcc'  # 176.000, 48%



init_word = 'aaaaaaabbbbbbbcc'  # 366.000, 67%
init_word = 'aaaaaabbbbbbbcc'  # 383.000, 57%
init_word = 'abcdefghi'  # 362.000, 75%



init_words = [
    #'aaaaaabbbbbbc',  # 10.000, 47%
    #'aaaabbbbccc',  # 10.000, 59%
    #'aaabbbcccd',  # 16.000, 66%
    #'aaabbccdd',   # 7.500, 66%
    #'abcdefgha',  # 181K, 74%
    #'abbbbbbbbbc',  # 110, 42%
    #'abbbbbbbbbbbbbbbbbc', # 630, 45
    #'abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbc', #1400, 47
    #'abbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
    #'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbc',  # 20.000, 49
    #'aabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbc',  # 27.000, 49
    #'aaaaaaaabbbbbbbcc', #774.00, 58%
    #'aabcdefgh', #180.000, 74%
    'aaabbcdef',  # 30.000, 71%

]


for init_word in init_words:
    print(init_word)
    alphabet = list(set(init_word))

    states, edges, G = generate_graph(init_word, nx_graph=True)
    #print(states)
    #print(edges)
    print('number of states: ', len(states))
    print(f'upper multinomial bound on number of states: {multinomial(len(init_word), int(len(init_word)/len(alphabet)), len(alphabet))}')
    print('number of edges: ', len(edges)/2)
    print(f'number of clique edges: {0.5*len(states)*(len(states)-1)}')
    print(f'cyclomatic number: {(0.5*len(edges) - len(states) +1)/(0.5*len(edges))}')
    #nx.draw(G, with_labels=True)
    #plt.show()
    print('\n')