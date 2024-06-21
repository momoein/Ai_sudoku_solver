import numpy as np

from csp.csp import *


# ___ problem modeling ___

def var_tuple(variable: str):
    return tuple([int(i) for i in variable.split("-")])

def variable_neighbors(variable:str, variables):
    i, j = var_tuple(variable)
    neighbors = []
    for v in variables:
        vi, vj = var_tuple(v)
        if v not in neighbors and (v != variable):
            if (vi == i or vj == j) or ((i) // 3 == (vi) // 3) and ((j) // 3 == (vj) // 3):
                neighbors.append(v)    
    return neighbors
        
def parse_neighbors(variables):
    neighbors = {}
    for var in variables:
        neighbors[var] = variable_neighbors(var, variables)
    return neighbors

def different_values_constraint(A, a, B, b):
    return a != b


def SudokuCSP(variables, domains):
    neighbors = parse_neighbors(variables)
    return CSP(variables, domains, neighbors, different_values_constraint)



def solver(sudoku_map):
    assignment = {f"{i}-{j}": sudoku_map[i, j] 
                for i in range(9) for j in range(9) if sudoku_map[i, j] != "_"}
    
    variables = [f"{i}-{j}" for i in range(9) for j in range(9)]
    
    domains = {var: list("123456789") for var in variables}

    problem = SudokuCSP(variables, domains)
    
    result = backtracking_search(problem, assignment, mrv, unordered_domain_values, forward_checking, probability=0.1)

    if result:
        table = np.zeros(shape=(9,9), dtype=str)
        for var in result:
            i, j = var_tuple(var)
            table[i, j] = result[var]
        return table
