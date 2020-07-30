

from we.arguments import Arguments
args = Arguments()
print('SMT converter', args.VARIABLES, args.ALPHABET)
if args.smt_solver == 'Z3':
    from z3 import *
    import os
    from pickle import Unpickler, Pickler
    import random


class WeToZ3(object):
    """
    Z3 implementation
    """
    def __init__(self, variables, alphabet, timeout, use_lc=False):
        self.alphabet = alphabet
        self.vars = variables
        self.solver = Solver()
        self.solver.set('timeout', timeout)
        self.solver.set('seed', 0)
        z3.set_param(verbose=10)

        self.use_lc = use_lc
        #self.solver.set('smt.string_solver', 'z3str3')

    def get_constant_chunk(self, word, position):
        chunk = ''
        letter = word[position]
        while letter in self.alphabet and position < len(word):
            chunk += letter
            position += 1
            if position < len(word):
                letter = word[position]
        return chunk

    def transform_pattern(self, w):
        var_last_chunk = True
        for i, letter in enumerate(w):
            if letter in self.vars:
                if i == 0:
                    side = String(letter)
                else:
                    side = Concat(side, String(letter))
                var_last_chunk = True
            else:
                if var_last_chunk:
                    constant_chunk = self.get_constant_chunk(word=w, position=i)
                    if i == 0:
                        side = StringVal(constant_chunk)
                    else:
                        side = Concat(side, StringVal(constant_chunk))
                    var_last_chunk = False
                else:
                    pass
        return side

    def transform_eq(self, eq):
        w1, w2 = eq.split('=')
        side1, side2 = self.transform_pattern(w1), self.transform_pattern(w2)
        self.solver.push()
        self.solver.add(side1 == side2)
        #self.solver.add(side2 == side1)

    def transform_lc(self, eq, num= 0):
        word = ''
        for x in self.vars:
            word += eq.coefficients_variables_lc[0][x]*x
        if len(word) == 0:
            return None
        for i, l in enumerate(word):
            if l in self.vars:
                if i== 0:
                    W= String(l)
                else:
                    W = Concat(W, String(l))
            elif l in self.alphabet:
                chunk = self.get_constant_chunk(word, position=i)
                if i == 0:
                    W = StringVal(chunk)
                else:
                    W =  Concat(W, StringVal(chunk))
        l_word = StringVal(eq.ell[num]*self.alphabet[0])
        self.solver.add(Length(W)>= Length(l_word))


    def transform(self, eq):
        self.transform_eq(eq.w)
        if self.use_lc:
            for _ in range(len(eq.coefficients_variables_lc)):
                self.transform_lc(eq, _)

    def eval(self, eq):
        self.transform(eq)
        out = self.solver.check()
        self.solver.reset()
        if out == sat:
            return 1.
        elif out == unsat:
            return -1.
        else:
            return 0.

if args.smt_solver == 'CVC4':
    import pysmt
    from pysmt.shortcuts import Solver, Equals, String, Symbol, StrConcat, StrLength, And
    from pysmt.typing import STRING
    from pysmt.logics import get_logic


    import os
    from pickle import Unpickler, Pickler
    import random

class WeToCVC4(object):
    """
    CVC4 implementation
    """
    def __init__(self, variables, alphabet, timeout, use_lc=False):
        self.alphabet = alphabet
        self.vars = variables
        self.problem = None
        self.solver = Solver(name='z3')
        self.solver.set_option('tlimit', timeout)
        self.use_lc = use_lc
        # self.solver.set('smt.string_solver', 'z3str3')

    def get_constant_chunk(self, word, position):
        chunk = ''
        letter = word[position]
        while letter in self.alphabet and position < len(word):
            chunk += letter
            position += 1
            if position < len(word):
                letter = word[position]
        return chunk

    def transform_pattern(self, w):
        var_last_chunk = True
        for i, letter in enumerate(w):
            if letter in self.vars:
                if i == 0:
                    side = Symbol(letter, STRING)
                else:
                    side = StrConcat(side,  Symbol(letter, STRING))
                var_last_chunk = True
            else:
                if var_last_chunk:
                    constant_chunk = self.get_constant_chunk(word=w, position=i)
                    if i == 0:
                        side = String(constant_chunk)
                    else:
                        side = StrConcat(side, String(constant_chunk))
                    var_last_chunk = False
                else:
                    pass
        return side

    def transform_eq(self, eq):
        w1, w2 = eq.split('=')
        side1, side2 = self.transform_pattern(w1), self.transform_pattern(w2)
        if self.problem is None:
            self.problem = Equals(side1, side2)
        #else:
        #    self.problem = And(self.problem, Equals(side1, side2))

    def transform_lc(self, eq, num=0):
        word = ''
        for x in self.vars:
            word += eq.coefficients_variables_lc[0][x] * x
        if len(word) == 0:
            return None
        for i, l in enumerate(word):
            if l in self.vars:
                if i == 0:
                    W = Symbol(l, STRING)
                else:
                    W = StrConcat(W, Symbol(l, STRING))
            elif l in self.alphabet:
                chunk = self.get_constant_chunk(word, position=i)
                if i == 0:
                    W = String(chunk)
                else:
                    W = StrConcat(W, String(chunk))
        l_word = String(eq.ell[num] * self.alphabet[0])
        self.problem = And(self.problem, StrLength(W) >= StrLength(l_word))
        # self.solver.add(StrLength(W)>= StrLength(l_word))

    def transform(self, eq):
        self.transform_eq(eq.w)
        if self.use_lc:
            for _ in range(len(eq.coefficients_variables_lc)):
                self.transform_lc(eq, _)

    def eval(self, eq):
        #eq.w='X=a'
        self.transform(eq)
        self.solver.add_assertion(self.problem)
        out = self.solver.solve(self.problem)
        #print(out)
        #print(self.solver.print_model())
        self.solver.exit()
        if out == True:
            return 1.
        elif out == False:
            return -1.
        else:
            return 0.

def SMT_eval(args, eq, timeout=None):
    try:
        timeout = int(args.timeout_time * 1000) if timeout is None else timeout
        if args.smt_solver == 'Z3':
            converter = WeToZ3(args.VARIABLES, args.ALPHABET, timeout, args.use_length_constraints)
        elif args.smt_solver == 'CVC4':
            converter = WeToCVC4(args.VARIABLES, args.ALPHABET, timeout, args.use_length_constraints)
        out = converter.eval(eq)
        return out
    except:
        try:
            timeout = 10000 if not args.active_tester else int(args.timeout_time * 1000)
            if args.smt_solver == 'Z3':
                converter = WeToZ3(args.VARIABLES, args.ALPHABET, timeout, args.use_length_constraints)
            elif args.smt_solver == 'CVC4':
                converter = WeToCVC4(args.VARIABLES, args.ALPHABET, timeout, args.use_length_constraints)
            out = converter.eval(eq)
            return out
        except:
            timeout = 10000 if not args.active_tester else int(args.timeout_time * 1000)
            if args.smt_solver == 'Z3':
                converter = WeToZ3(args.VARIABLES, args.ALPHABET, timeout, args.use_length_constraints)
            elif args.smt_solver == 'CVC4':
                converter = WeToCVC4(args.VARIABLES, args.ALPHABET, timeout, args.use_length_constraints)
            out = converter.eval(eq)
            return out

if __name__ == '__main__':
    w = 'Xa=aX'
    converter = WeToZ3(['X'],['a'],100, False)
    converter.transform_eq(w)
    print(converter.problem)
