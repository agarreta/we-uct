import re
from collections import OrderedDict
from random import shuffle

class WordEquationTransformations(object):

    # todo: throughout all project I do a lot of split('='). Optimize?

    def __init__(self, args):

        self.args = args

    def normal_form(self, eq, minimize=True, mode='play'):
        if self.args.use_normal_forms:
            if minimize is None:
                minimize = not self.args.use_true_symmetrization
            # print(self.eq.lp['abelian_form'].items())
            if self.args.automatic_compress and mode != 'generation':
                new_eq = ''
                while new_eq != eq.w:
                    new_eq = eq.w
                    eq = self.compress(eq)

            if (self.args.generation_mode != 'constant_side') and (not self.args.quadratic_mode):
                new_w_split = eq.w.split('=')
                new_w_split.sort()
                new_w = new_w_split[0] + '=' + new_w_split[1]
                eq.w = new_w

            if self.args.use_length_constraints:
                eq = self.treat_lc(eq, mode)

            if mode != 'generation':
                eq = self.del_pref_suf(eq)

            if minimize and self.args.use_normal_forms:
                auto = self.get_automorphism(eq)
                eq = self.apply_automorphism(eq, auto, mode)
            return eq
        else:
            if mode != 'generation':
                eq = self.del_pref_suf(eq)

            if self.args.use_length_constraints:
                for i,x in enumerate(eq.coefficients_variables_lc):
                    for _ in range(len(x)):
                        eq = self.treat_lc(eq, mode,_,i)
            return eq

            def normal_form_large_version(w, alphabet, variables, num_digits):

                def substitution_fun(char, auto, size_auto, letter_list):
                    if char not in auto.keys():
                        new_char = letter_list[size_auto]
                        auto[char] = new_char
                        size_auto += 1
                    else:
                        new_char = auto[char]
                    return new_char, auto, size_auto

                alph_auto = {}
                size_alph_auto = 0
                var_auto = {}
                size_var_auto = 0
                new_w = ''
                pointer = 0
                while pointer < len(w):
                    char = w[pointer]
                    if char in alphabet:
                        pointer += 1
                        new_char, alph_auto, size_alph_auto = substitution_fun(char, alph_auto, size_alph_auto,
                                                                               alphabet)
                        new_w += new_char
                    elif char == '=':
                        pointer += 1
                        new_w += '='
                    else:
                        char = w[pointer: pointer + num_digits]
                        pointer += num_digits
                        new_char, var_auto, size_var_auto = substitution_fun(char, var_auto, size_var_auto, variables)
                        new_w += new_char
                return new_w

            eq.w = normal_form_large_version(eq.w, self.args.ALPHABET, self.args.VARIABLES, self.args.num_var_digits_large_mode)

        return eq

    def compress(self, eq):
        if self.args.ALPHABET[-1]  in eq.w:
            return eq
        left_allowed = {letter: True for letter in self.args.ALPHABET[:-1]}
        right_allowed = {letter: True for letter in self.args.ALPHABET[:-1]}
        for letter in self.args.ALPHABET:
            split = eq.w.split(letter)
            left_split = [x[-1:] for x in split[:-1]]  # symbols on the left of letter
            right_split = [x[:1] for x in split[1:]]
            if any([x in self.args.VARIABLES for x in left_split]):
                right_allowed[letter] = False
            if any([x in self.args.VARIABLES for x in right_split]):
                left_allowed[letter] = False
        allowed_tuples = [l1 + l2 for l1, val1 in left_allowed.items() if val1 for l2, val2 in right_allowed.items() if
                          val2]
        if len(allowed_tuples) == 0:
            eq.attempted_wrong_move = True
            return eq
        else:
            counts = [[eq.w.count(pair), pair] for pair in allowed_tuples]
            if all([x == 0 for x in counts]):
                eq.attempted_wrong_move = True
                return eq
            else:
                counts.sort()
                new_w = re.sub(counts[-1][-1], self.args.ALPHABET[-1], eq.w)
                eq.w = new_w
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
        if False:
            char_auto = re.sub( '[' + ''.join(self.args.VARIABLES + ['.', '=']) + ']', '' , order)
            var_auto = re.sub('[' + ''.join(self.args.ALPHABET + ['.', '=']) + ']', '', order)

            for x in self.args.ALPHABET:
                if x not in char_auto:
                    char_auto += x

            for x in self.args.VARIABLES:
                if x not in var_auto:
                    var_auto += x

        automorphism = {'.':'.', '=':'='}
        automorphism.update({var_auto[i]: self.args.VARIABLES[i] for i in range(len(var_auto))})
        automorphism.update({char_auto[i]: self.args.ALPHABET[i] for i in range(len(char_auto))})
        return automorphism

    def apply_automorphism_to_dictionary(self,dictionary, auto):
        return {auto[key]: value for key,value in dictionary.items()}

    def apply_automorphism(self, eq, auto, mode = 'play'):
        eq.w = eq.w.translate(str.maketrans(auto))
        if self.args.use_length_constraints:
            eq.weights_constants_lc = self.apply_automorphism_to_dictionary(eq.weights_constants_lc, auto)
            # { auto[key] : value for key, value in eq.weights_constants_lc.items() }
            eq.coefficients_variables_lc = self.apply_automorphism_to_dictionary(eq.coefficients_variables_lc, auto)
            # { auto[key] : value for key, value in eq.coefficients_variables_lc.items() }
            if mode == 'generation':
                eq.num_letters_per_variable = { auto[key]: self.apply_automorphism_to_dictionary(value, auto) for
                                                key, value in eq.num_letters_per_variable.items() }
        # eq.candidate_sol = { auto[key[0]] + key[1]: value for key, value in eq.candidate_sol.items() }
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


    def treat_lc(self, eq, mode = 'play', num=0, i=0):
        # if eq.ell <= 0:
        # if mode == 'generation':
        #     if all([eq.coefficients_variables_lc[x] <= 0 for x in self.args.VARIABLES]):
        #         for x in self.args.VARIABLES:
        #             # todo: a bit unefficient since we checked <= for all vars before. improve?
        #             if eq.coefficients_variables_lc[x] < eq.ell:
        #                 if x in eq.w:
        #                     eq.w = re.sub(x, '', eq.w)
        #                     eq.coefficients_variables_lc[x] = 0
        #                     # eq.lp['abelian_form'][x] = 0
        #             eq.ell = 0
        #         # return eq
        #
        #     elif all([eq.coefficients_variables_lc[x] >= 0 for x in self.args.VARIABLES]):
        #         if eq.ell <= 0:
        #             eq.nullify_lc()
        #             # return eq
        #
        #     elif all([eq.coefficients_variables_lc[x] == 0 for x in self.args.VARIABLES]):
        #         eq.ell = 0
        #         # return eq
        #
        #
        # else:
        for x in self.args.VARIABLES:
            try:
                coef = eq.coefficients_variables_lc[i][num][x]
            except:
                a=1
            if coef != 0:
                if x not in eq.w:
                    if coef > 0:
                        eq.nullify_lc(num,i)
                    elif coef < 0:
                        eq.coefficients_variables_lc[i][num][x] = 0

        #for x in self.args.ALPHABET:
        #    coef = eq.weights_constants_lc[x]
        #    if coef != 0:
        #        if x not in eq.w:
        #            eq.weights_constants_lc[x] = 1

        return eq
            # else:
            #     for x in self.args.VARIABLES:
            #         if eq.coefficients_variables_lc[x] > eq.ell:
            #             #
            #             eq.coefficients_variables_lc[x] = eq.ell + 1


    # def treat_one_side_is_var(self, eq):
    #
    #     def main_subfunction(eq, var, subst):
    #         assert len(var_side) == 1
    #         eq.ell = eq.ell - eq.coefficients_variables_lc[var] * self.count_ctts(subst)
    #         eq.coefficients_variables_lc.update({
    #             x: eq.coefficients_variables_lc[x] + eq.coefficients_variables_lc[var] * subst.count(x) for x in
    #             self.args.VARIABLES if x != var
    #         })
    #         eq.coefficients_variables_lc.update({
    #             var: 0
    #         })
    #         if len(eq.not_normalized) == 0:
    #             eq.not_normalized = '-treated case var=...'
    #         else:
    #             eq.not_normalized += '-treated case var=...'
    #         return eq
    #
    #
    #     w_split = eq.w.split('=')
    #     if w_split[0] in self.args.VARIABLES and self.utils.is_constant(w_split[1]):
    #         eq = main_subfunction(eq, w_split[0], w_split[1]):
    #     elif w_split[1] in self.args.VARIABLES and self.utils.is_constant(w_split[0]):
    #         eq = main_subfunction(eq, w_split[1], w_split[0]):
    #
    #     return eq
    #
    # def treat_case_variable_side(self, eq, var_side, other_side):
    #     assert len(var_side) == 1
    #     # if var_side in other_side:
    #     #     # then we have e.g. X=aXbY
    #     #     return False
    #     # else:
    #     # then we have e.g. X=aYbZ
    #     eq = self.update_LC_with_var_subtitution(eq, var_side, other_side)
    #     # if True:
    #     if self.LC_is_sat(eq):
    #         eq.sat = 1
    #         return True
    #     else:
    #         eq.sat = -1
    #         return False