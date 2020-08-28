
from string import ascii_lowercase, ascii_uppercase
import os
import re
num_vars =10
num_alph = 6
alphabet = [x for x in ascii_lowercase]
vars_ = [x for x in ascii_uppercase[::-1]] + [x for x in '0123456789'] 
vars_=vars_


def get_constant_chunk(word, position):
    chunk = ''
    letter = word[position]
    while letter in alphabet and position < len(word):
        chunk += letter
        position += 1
        if position < len(word):
            letter = word[position]
    return chunk

def transform_pattern(w):
    var_last_chunk = True
    for i, letter in enumerate(w):
        if letter in vars_:
            if i == 0:
                side = letter
            else:
                side = f'{side} {letter}'
            var_last_chunk = True
        else:
            if var_last_chunk:
                constant_chunk = get_constant_chunk(word=w, position=i)
                if i == 0:
                    side = f'"{constant_chunk}"'
                else:
                    side = f'{side} "{constant_chunk}"'
                var_last_chunk = False
            else:
                pass
    return side


def transform_eq_inner(eq):
    if len(eq.split('='))==1:
        print('ERROR', eq, '0.8s')
        assert False
    w1, w2 = eq.split('=') # holis
    side1, side2 = transform_pattern(w1), transform_pattern(w2)
    side1 = f'(str.++ {side1})' if len(side1.split(' ')) > 1 else side1
    side2 = f'(str.++ {side2})' if len(side2.split(' ')) > 1 else side2
    problem = f'\n(assert (= {side1}  {side2}))'
    return problem
    
    
def transform_lc(lc):
	slc=''
	lc = lc.split(':')[1].split('|')[:-1]
	for x in lc:
		main, ell = x.split('>')[1]
		var = main[-1]
		coef=main[:-1]
		slc+=f'\n(assert (>=(* (str.len {var}) {coef}) {ell}))'
def transform_eq(e):
    if  '>' in e:
        w=e.split(':')[0]
    else:
        w=e
    local_VARS = set({x for x in w if x in vars_})
    local_ALPH = set({x for x in w if x in alphabet})
    smt_problem = '(set-logic QF_S)'
    for x in local_VARS:
        smt_problem += f'\n(declare-fun {x} () String)'
    smt_problem += transform_eq_inner(w)
    if  '>' in e:
        smt_problem += transform_lc(e)
    smt_problem += '\n(check-sat)'

    with open('word_equation.smt', 'w+') as f:
        f.write(smt_problem)


    
with open('word_equation.txt', 'r') as f:
	eq = f.read()
   
with open('timeout.txt', 'r') as f:
	t = int(f.read())
transform_eq(eq)
stream = os.popen(f'./cvc4 --lang smt --seed=0 --tlimit={t} word_equation.smt')
output = stream.read()
print(output)







