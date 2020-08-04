import re
from collections import OrderedDict
from random import shuffle

class WordEquationTransformations(object):
    def __init__(self, args):
        self.args = args

    def normal_form(self, eq, mode='play'):
        if self.args.use_normal_forms:
            #if (self.args.generation_mode != 'constant_side') and (not self.args.quadratic_mode):
            #    new_w_split = eq.w.split('=')
            #    new_w_split.sort()
            #    new_w = new_w_split[0] + '=' + new_w_split[1]
            #    eq.w = new_w

            if mode != 'generation':
                eq = self.del_pref_suf(eq)

            if self.args.use_normal_forms:
                auto = self.get_automorphism(eq)
                eq = self.apply_automorphism(eq, auto, mode)
            return eq
        else:
            if mode != 'generation':
                eq = self.del_pref_suf(eq)
            return eq

    def get_automorphism(self, eq, type='canonical'):

        if type == 'canonical':
            order = ''.join(OrderedDict.fromkeys(eq.w).keys())
        else:
            order = list(set(eq.w))
            shuffle(order)
            order = ''.join(order)
        order = re.sub('=','',order)
        order = re.sub('\.','',order)
        var_auto =''
        char_auto = ''
        for x in order:
            if x.isupper():
                var_auto += x
            else:
                char_auto += x
        automorphism = {'.':'.', '=':'='}
        automorphism.update({var_auto[i]: self.args.VARIABLES[i] for i in range(len(var_auto))})
        automorphism.update({char_auto[i]: self.args.ALPHABET[i] for i in range(len(char_auto))})
        return automorphism

    def apply_automorphism_to_dictionary(self,dictionary, auto):
        return {auto[key]: value for key,value in dictionary.items()}

    def apply_automorphism(self, eq, auto):
        eq.w = eq.w.translate(str.maketrans(auto))
        return eq

    def del_pref_suf(self, eq):
        """note this does not change the LP (abelian form)"""

        def length_longest_common_prefix(strs):

            if not strs: return 0
            shortest_str = min(strs, key=len)
            length = 0
            for i in range(len(shortest_str)):
                if all([x.startswith(shortest_str[:i + 1]) for x in strs]):
                    length += 1
                else:
                    break

            return length

        def length_longest_common_sufix(strs):

            if not strs: return 0
            shortest_str = min(strs, key=len)
            length = 0
            for i in range(len(shortest_str)):
                if all([x.endswith(shortest_str[-i - 1:]) for x in strs]):
                    length += 1
                else:
                    break

            return length

        w_split = eq.w.split('=')
        l_prefix = length_longest_common_prefix(w_split)
        w_l = w_split[0][l_prefix:]
        w_r = w_split[1][l_prefix:]

        l_suf = length_longest_common_sufix([w_l, w_r])
        if l_suf > 0:
            w_l = w_l[:-l_suf]
            w_r = w_r[:-l_suf]

        if len(w_l) == 0:
            w_l = '.'
        if len(w_r) == 0:
            w_r = '.'

        eq.w = w_l + '=' + w_r
        return eq
