import pandas as pd

def read_knapsack_data_from_csv(csv_file):
    df = pd.read_csv(csv_file, index_col=[0, 1])
    datasets = []
    for name, group in df.groupby(level=0):
        dataset = {
            'weights': group['weights'].tolist(),
            'values': group['values'].tolist(),
            'capacity': group['capacity'].iloc[0],
            'distribution': group['distribution'].iloc[0]
        }
        datasets.append(dataset)
    return datasets

def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][capacity]

def solve_knapsack_and_average(csv_file):
    datasets = read_knapsack_data_from_csv(csv_file)
    total_value = 0
    
    for dataset in datasets:
        value = knapsack(dataset['weights'], dataset['values'], dataset['capacity'])
        total_value += value
    
    average_value = total_value / len(datasets) if datasets else 0
    return average_value

# Usage
csv_file = './funsearch/funsearch/knapsack_datasets.csv'
average_value = solve_knapsack_and_average(csv_file)
print(f"The average value of all solved knapsack problems is: {average_value}")
