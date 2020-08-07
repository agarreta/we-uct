from .word_equation_generator_quadratic import WordEquationGeneratorQuadratic
from .word_equation_utils import WordEquationUtils, seed_everything
from .word_equation_transformations import WordEquationTransformations
from .word_equations_moves_quadratic import WordEquationMovesQuadratic
from copy import deepcopy

class WE(object):
    def __init__(self, args, seed=None):
        self.args = args
        if seed is not None:
            seed_everything(seed)
        self.utils = WordEquationUtils(args, seed)
        self.transformations = WordEquationTransformations(args)
        self.moves = WordEquationMovesQuadratic(args, seed)
        self.generator = WordEquationGeneratorQuadratic(args, seed)