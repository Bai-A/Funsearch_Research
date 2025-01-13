import numpy as np

def generate_items_and_capacity():
    """Generates items and a bin capacity with a fixed seed for repeatability."""
    np.random.seed(42)  # Set a fixed seed for reproducibility
    num_items = 40  # Generate number of items between 5 and 20
    items = np.random.randint(1, 100, size=num_items)  # Generate item sizes between 1 and 10
    bin_capacity = 150  # Generate bin capacity between 10 and 20
    return items, bin_capacity



from ortools.linear_solver import pywraplp

def solve_bin_packing(items, bin_capacity):
    """Solves the bin packing problem to find the minimum number of bins required."""
    # Create the solver using the SCIP backend
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return None

    n = len(items)  # Number of items
    max_bins = n  # Maximum number of bins required would be one item per bin

    # Variables
    # x[i, j] = 1 if item i is in bin j
    x = {}
    for i in range(n):
        for j in range(max_bins):
            x[i, j] = solver.BoolVar(f'x_{i}_{j}')

    # y[j] = 1 if bin j is used
    y = [solver.BoolVar(f'y_{j}') for j in range(max_bins)]

    # Constraints
    # Each item must be in exactly one bin
    for i in range(n):
        solver.Add(solver.Sum(x[i, j] for j in range(max_bins)) == 1)

    # Bin capacity constraint
    for j in range(max_bins):
        solver.Add(solver.Sum(items[i] * x[i, j] for i in range(n)) <= bin_capacity * y[j])

    # Linking number of bins and items
    for i in range(n):
        for j in range(max_bins):
            solver.Add(x[i, j] <= y[j])

    # Objective: minimize the number of bins used
    solver.Minimize(solver.Sum(y[j] for j in range(max_bins)))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Minimum number of bins required:', solver.Objective().Value())
        for j in range(max_bins):
            if y[j].solution_value() > 0.5:  # Bin is used
                print(f'Bin {j+1} contains items with sizes:', [items[i] for i in range(n) if x[i, j].solution_value() > 0.5])
    else:
        print('The problem does not have an optimal solution.')

items, bin_capacity = generate_items_and_capacity()
print(items)
solve_bin_packing(items, bin_capacity)


