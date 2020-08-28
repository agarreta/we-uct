*NOTE:* Section under construction

This folder contains a python script for each specific solver to be used.
Each script is executed on the terminal by the MCTS_solver algorithm each time it needs to call 'solver'. 
The script then transforms a given equation into a suitable form and then calls the solver on the transformed equation through a terminal command.
Each script is to be placed in the folder from where one would run the soler (default directories are listed below). 
Two reasons for operating like this are: 
1) Not all solvers have a python API but nevertheless are easily callable from the terminal. This allows to use any solver which is currently working on the system.
2) For tose that do have a python API, calling the solver directly with our python interpreter produced all sorts of unexpected behaviors. 
3) Furthermore, each API has a syntax particular to the solver, which can be quite cumbersome to learn. 
