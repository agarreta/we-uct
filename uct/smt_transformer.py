import re
import os
from pickle import Pickler
from uct.we.word_equation import WordEquation
from we.arguments import Arguments
from string import ascii_uppercase, ascii_lowercase

args = Arguments()
args.VARIABLES = list(ascii_uppercase)
args.ALPHABET = list(ascii_lowercase)


def read_file(path):
    file = open(path, 'r')
    return file.read()

def transform(folder, save_to):
    """
    Transforms all smt files (each containing a word equation) in a folder into a list of strings, each one a word equation, e.g. 'Xa=aX'.
    Tuning the function may be necessary depending on the exact formatting of the smt files.

    folder: folder containing the smt files
    save_to: folder to save the output list
    """
    eqs = []
    while len(eqs) < len(folder):
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

    with open(save_to, 'wb+') as f:
        Pickler(f).dump(eqs)

if __name__ == '__main__':
    folder = f'benchmarks/03_track/smt'
    transform(folder, folder)
