import re
import os
from pickle import Pickler
from we.word_equation.word_equation import WordEquation
from we.arguments import Arguments
from string import ascii_uppercase, ascii_lowercase
from copy import deepcopy

args = Arguments()
args.VARIABLES = list(ascii_uppercase)
args.ALPHABET = list(ascii_lowercase)

track = 3

def read_file(path):
    file = open(path, 'r')
    return file.read()

#os.chdir('test')
folder = f'benchmarks/0{track}_track/smt'
eqs = []
it=0
while len(eqs) < 200:
    for i, file in enumerate(os.listdir(folder)):
        f = read_file(os.path.join(folder, file))
        f = f.split('\n')[1:-3]
        print(f)
        left_sides = []
        right_sides = []
        list_coef_var_dicts=[{var: 0 for var in args.VARIABLES}]
        list_ells=[0]
        for clause in f:
            if 'assert (=' in clause:
                c = clause.split('str.++ ')[1:]
                c1 = c[0].split(')')[0]
                c2 = c[1].split(')')[0]
                c1 = re.sub('"', '', c1)
                c1 = re.sub(' ', '', c1)
                c2 = re.sub('"', '', c2)
                c2 = re.sub(' ', '', c2)
                left_sides.append(c1)
                right_sides.append(c2)
            if 'assert (<=' in clause:
                c_split = clause.split(')')
                var_main = c_split[0][-1]
                coef = -int(c_split[1])
                coef_var_dict = {var: 0 for var in args.VARIABLES}
                coef_var_dict[var_main] = coef
                ell = -int(c_split[2])
                list_coef_var_dicts.append(coef_var_dict)
                list_ells.append(ell)
            if 'assert (>=' in clause:
                c_split = clause.split(')')
                var_main = c_split[0][-1]
                coef = int(c_split[1])
                coef_var_dict = {var: 0 for var in args.VARIABLES}
                coef_var_dict[var_main] = coef
                ell = int(c_split[2])
                list_coef_var_dicts.append(coef_var_dict)
                list_ells.append(ell)
        eq=''
        for x in left_sides:
            eq += x + '+'
        eq = eq[:-1]
        eq += '='
        for x in right_sides:
            eq += x + '+'
        eq = eq[:-1]

        print(eq+'\n')
        eq = WordEquation(args,w=eq)
        eq.coefficients_variables_lc = list_coef_var_dicts
        eq.ell = list_ells
        eqs.append(eq)
    if it == 0:
        print(len(eqs), len(os.listdir(folder)))
        assert len(eqs) == len(os.listdir(folder))
    it+=1
    print(it, len(eqs), len(os.listdir(folder)))



with open(f'benchmarks/0{track}_track/transformed', 'wb+') as f:
    Pickler(f).dump(eqs)

