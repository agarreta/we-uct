from we.SMTSolver import WeToCVC4
from pickle import Unpickler
from we.arguments import Arguments
import re


def transform_eq(w):
    local_VARS = set({x for x in w if x in VARS})
    local_ALPH = set({x for x in w if x in ALPH})
    conv = WeToCVC4(local_VARS, local_ALPH, 10000, False)
    conv.transform_eq(w)
    smt_problem = '(set-logic QF_S)\n'
    for x in local_VARS:
        smt_problem += f'(declare-fun {x} () String)\n'
    pblm = re.sub('str\.\+\+\(', '', str(conv.problem))
    pblm = re.sub('\)', '', pblm)

    side1, side2 =  pblm.split('=')
    smt_problem += f'(assert (= (str.++ {side1[1:]}) (str.++ {side2}) ))\n'
    smt_problem += '(check-sat)\n(get-model)'
    smt_problem = re.sub('\,', '', smt_problem)

    return smt_problem

def transform_eq_woorpje(w):
    local_VARS = set({x for x in w if x in VARS})
    local_ALPH = set({x for x in w if x in ALPH})
    conv = WeToCVC4(local_VARS, local_ALPH, 10000, False)
    conv.transform_eq(w)
    smt_problem = 'Variables {'
    for x in local_VARS:
        smt_problem += x
    smt_problem += '}\nTerminals {'
    for x in local_ALPH:
        smt_problem += x
    smt_problem += '}\nEquation: '
    smt_problem += w.split('=')[0] + ' = ' + w.split('=')[1]
    smt_problem += '\nSatGlucose(100)'

    return smt_problem




def save_file(eq_smt, filepath):
    with open(filepath, 'w+') as f:
        f.write(eq_smt)

def transform_pool(filepath):
    with open(filepath, 'rb') as f:
        pool = Unpickler(f).load()
    for i, eq in enumerate(pool):
        eq_smt = transform_eq(eq.w)
        print(f'"{eq.w}",')
        save_path = f'test/benchmarks/quadratic_smt/{i}.eq'
        save_file(eq_smt, save_path)

if __name__ == '__main__':
    args = Arguments()
    args.change_parameters_by_folder_name('we_alpha0_disc90_smaller_seed2')
    VARS = args.VARIABLES
    ALPH = args.ALPHABET
    print(VARS, ALPH)

    transform_pool('test/benchmarks/pool_lvl_35_26_size_300_quadratic-oriented_tiny.pth.tar')
