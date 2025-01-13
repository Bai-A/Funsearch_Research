from ortools.linear_solver import pywraplp
import numpy as np
def solve_optimal_knapsack(items, values, capacity):
    """Solves the knapsack problem to find the maximum value that can be achieved within the given capacity."""
    # Create the solver with the SCIP backend, suitable for integer programming
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP solver is not available.")
        return 0

    num_items = len(items)
    
    # Decision variables: x[i] is 1 if item i is included in the knapsack, otherwise 0
    x = [solver.BoolVar(f'x_{i}') for i in range(num_items)]

    # Objective: Maximize the total value of the knapsack
    objective = solver.Objective()
    for i in range(num_items):
        objective.SetCoefficient(x[i], float(values[i]))  # Ensure coefficient is a float
    objective.SetMaximization()

    # Constraint: The total size of items in the knapsack should not exceed the capacity
    size_constraint = solver.Constraint(-solver.infinity(), float(capacity))  # Ensure capacity is a float
    for i in range(num_items):
        size_constraint.SetCoefficient(x[i], float(items[i]))  # Ensure weights are treated as floats

    # Solve the problem
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Solution:")
        print("Objective value =", objective.Value())
        total_weight = sum(x[i].solution_value() * items[i] for i in range(num_items))
        print("Total weight of items =", total_weight)
        selected_items = [i for i in range(num_items) if x[i].solution_value() > 0.5]
        print("Selected items:", selected_items)
        return objective.Value()
    else:
        print("The problem does not have an optimal solution.")
        return 0

# Example setup
np.random.seed(49)
num_items = 40
items = np.random.randint(1, 100, size=num_items)  # Item weights
values = np.random.randint(1, 100, size=num_items)  # Item values
capacity = 150  # Knapsack capacity

# Find the optimal solution
max_value = solve_optimal_knapsack(items, values, capacity)
print(items,values)
print("Maximum value in knapsack (Optimal):", max_value)
