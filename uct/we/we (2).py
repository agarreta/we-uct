from .word_equation_generator import WordEquationGenerator
from .word_equation_moves import WordEquationMoves
from .word_equation_utils import WordEquationUtils
from .word_equation_transformations import WordEquationTransformations

class WE(object):
    def __init__(self, args):
        self.args = args
        self.utils = WordEquationUtils(args)
        self.moves = WordEquationMoves(args)
        self.transformations = WordEquationTransformations(args)
        self.generator = WordEquationGenerator(args)


