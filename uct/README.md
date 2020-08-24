Description of files:
  - Main files:
    - arcade_train.py: class for training the network in MCTS_nn as explained in the paper
    - arcade_test.py: class for testing algorithms on a given pool of equations
    - player.py:  used by the previous two
    - mcts.py: used in player.py to obtain a policy according to the MCTS algorithm
    - newrensnet.py: contains the network used in MCTS_nn
    - neural_net_wrapper.py: class for managing and training the network above
    - SMT_solver.py: class handling different string solvers
    - we (folder): files with classes corresponding to the MDP used in our algorithms.

  - Utility files:
    - arguments.py: config file
    - utils.py
    - smt_transformer.py: functions for transforming a given equation in smt format into a WE object used in our code. May require tunning depending on the exact formatting of the smt input file

  
  
