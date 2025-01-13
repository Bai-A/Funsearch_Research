"""Finds largest amount of value that can be carried.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import numpy as np
import funsearch
import pandas as pd
@funsearch.run
def evaluate(n) -> int:
  """Generates a random knapsack problem with a fixed seed, solves it, and returns the maximum value."""
  results = solve()
  max_value = sum(results)/len(results)
  return int(max_value)

def solve(csv_path = './funsearch/funsearch/knapsack_datasets.csv'):
  """Reads knapsack instances from a CSV file, applies a knapsack algorithm using a priority function, and returns the list of maximum values."""
  # Read CSV and split into separate instances based on index
  df = pd.read_csv(csv_path, index_col=[0, 1])
  results = []

  for _, group in df.groupby(level=0):  # Group by the instance index
      weights = group['weights'].values
      values = group['values'].values
      capacity = group['capacity'].iloc[0]
      
      # Apply knapsack algorithm with priority
      items_with_priority = [(weights[i], values[i], priority(weights[i], values[i], capacity)) for i in range(len(weights))]
      items_with_priority.sort(key=lambda x: x[2], reverse=True)  # Sort by priority, high to low
      
      total_value = 0
      current_size = 0
      
      for size, value, _ in items_with_priority:
          if current_size + size <= capacity:
              total_value += value
              current_size += size
      
      results.append(total_value)

  return results

@funsearch.evolve
def priority(size: int, value: int, capacity: int) -> float:
  """Returns the priority with which we want to add an item to the knapsack, based on value density."""
  return 0.0  # Value per unit size, higher density means higher priority

