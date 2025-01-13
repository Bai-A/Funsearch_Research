"""Finds smallest amount of bins.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
"""
import numpy as np
import funsearch

@funsearch.run
def evaluate(n: int) -> int:
  """Generates a random bin packing problem with a fixed seed, solves it, and saves the result to a file."""
  bins = solve()
  return -len(bins)

def solve() -> list[list[int]]:
  """Generates random items and bin capacity with a fixed seed, then packs items into bins based on priority and saves the data to a text file."""
  np.random.seed(42)  # Set a fixed seed for repeatability
  num_items = 40  # Randomly choose between 5 and 20 items
  items = np.random.randint(1, 100, size=num_items)  # Item sizes between 1 and 10
  bin_capacity = 150  # Bin capacity between 10 and 20
    
  # Calculate priority for each item and sort items based on priority
  items_with_priority = [(item, priority(item, bin_capacity)) for item in items]
  items_with_priority.sort(key=lambda x: x[1], reverse=True)  # Higher priority items first
  sorted_items = [item[0] for item in items_with_priority]

  bins = []
  for item in sorted_items:
    placed = False
    for bin in bins:
          if sum(bin) + item <= bin_capacity:
                bin.append(item)
                placed = True
                break
    if not placed:
            bins.append([item])  # Start a new bin if no existing bin can accommodate the item

  return bins

@funsearch.evolve
def priority(item: int, bin_capacity: int) -> float:
  """Returns the priority with which we want to add an item to the bins.
  In this simple heuristic, larger items get higher priority."""
  return 0.0 # Prioritize larger items based on their size relative to bin capacity

print(evaluate(0))