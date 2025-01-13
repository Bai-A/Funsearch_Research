"""This is an inv management problem. Finds best policy for the lowest cost.
On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
Dont add any more comments
"""
from scipy.optimize import minimize
import or_gym
import numpy as np
import funsearch
@funsearch.run
def evaluate(n) -> int:
  results = solve()
  max_value = sum(results)/len(results)
  return int(max_value)




def solve():
  # Register environment
  def dfo_func(policy, env, *args):
    '''
    Runs an episode based on current base-stock model 
    settings. This allows us to use our environment for the 
    DFO optimizer.
    '''
    env.reset() # Ensure env is fresh
    rewards = []
    done = False
    while not done:
        action = priority(policy, env)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
            
    rewards = np.array(rewards)
    
    # Return negative of expected profit
    return -1 / env.num_periods * np.sum(  rewards)
  
  def optimize_inventory_policy(env_name, fun,
    init_policy=None, env_config={}, method='Powell'):
    
    env = or_gym.make(env_name, env_config=env_config)
    
    if init_policy is None:
        init_policy = np.ones(env.num_stages-1)
        
    # Optimize policy
    out = minimize(fun=fun, x0=init_policy, args=env, 
        method=method)
    
    policy = out.x.copy()
    
    # Policy must be positive integer
    policy = np.round(np.maximum(policy, 0), 0).astype(int)
    
    return policy, out
  env_name='InvManagement-v1'
  env_config = {}
  policy, out = optimize_inventory_policy('InvManagement-v1',
    dfo_func,init_policy=priority(None,or_gym.make(env_name, env_config={})))
  print("Re-order levels: {}".format(policy))

  env = or_gym.make(env_name, env_config=env_config)
  eps = 1000
  rewards = []
  for i in range(eps):
      env.reset()
      reward = 0
      while True:
          action = priority(policy, env)
          s, r, done, _ = env.step(action)
          reward += r
          if done:
              rewards.append(reward)
              break
  return rewards

@funsearch.evolve
def priority(policy, env):
  '''
  **Objective:**
  Design an innovative and unconventional inventory policy that significantly improves the performance score of this inventory management problem. The policy should not be based on popular or well-known algorithms. Instead, explore novel ideas, unique mathematical relationships, or interdisciplinary approaches that are rarely applied in inventory management.

  **Environment Details:**
  - **env.p**: [positive float] Unit price for the final product.
  - **env.r**: [non-negative float; dimension |Stages|] Unit cost for replenishment orders at each stage.
  - **env.k**: [non-negative float; dimension |Stages|] Backlog cost or goodwill loss (per unit) for unfulfilled orders.
  - **env.h**: [non-negative float; dimension |Stages|-1] Unit holding cost for excess on-hand inventory at each stage.
  - **env.c**: [positive integer; dimension |Stages|-1] Production capacities for each supplier.
  - **env.L**: [non-negative integer; dimension |Stages|-1] Lead times between stages.
  - **env.I**: On-hand inventory at the start of each period at each stage (except last one).
  - **env.T**: Pipeline inventory at the start of each period at each stage (except last one).
  - **env.D**: Customer demand at each period at the retailer.
  - **env.B**: Backlog at each period at each stage.

  **Guidelines:**
  - **Leverage Underutilized Data**: Utilize information from the environment that is often neglected in standard models.
  - **Innovative Concepts**: Incorporate concepts from fields such as chaos theory, fractals, quantum computing principles, or biologically inspired algorithms like genetic algorithms or neural networks.
  - **Dynamic Adaptation**: Allow the policy to adapt dynamically to changing conditions within the environment.
  - **Interdisciplinary Approaches**: Apply mathematical techniques or theories not traditionally used in inventory management.

  **Constraints:**
  - The actions returned should be a vector of integers indicating the number of units to order at each stage (same shape as `policy`).
  - You may change the shape of `policy`, but you need to produce an `unc_actions` vector that matches the shape of `env.c` as output.
  - Ensure that actions are feasible within the environment's constraints using the following code:
    ```python
    inv_const = np.hstack([env.I[env.period, 1:], np.Inf])
    actions = np.minimum(env.c, np.minimum(unc_actions, inv_const))
    ```

  **Examples of Innovative Approaches:**
  - **Agent-Based Modeling**: Model each stage as an agent with its own decision-making process.
  - **Complex Network Analysis**: Analyze the supply chain as a complex network to identify optimal ordering strategies.
  - **Entropy-Based Methods**: Use entropy measures to determine uncertainty and adjust orders accordingly.
  - **Fuzzy Logic**: Incorporate fuzzy logic to handle uncertainty in demand and supply.

  **Implementation Notes:**
  - Avoid using standard policies like base-stock levels or (s, S) policies.
  - Clearly comment your code to explain the unconventional methods used.
  - Focus on creating a policy that is not only unique but also improves performance.

  **Action Items:**
  - Implement your innovative policy below.
  - Ensure that your code is efficient and avoids unnecessary complexity.


  # Your innovative policy implementation starts here

  # Example of starting with an unconventional approach:
  # Let's suppose we use a simple form of a genetic algorithm concept
  # to adjust order quantities based on the performance of previous periods.

  # Initialize unc_actions vector

  # Implement your unconventional algorithm
  # For illustration purposes, here is a placeholder for your code:
  # unc_actions = your_unconventional_method(env, policy)

  # Ensure actions are within feasible bounds

  # if policy is None, return the shape of the policy  
  # you should always use this first to specify the shape of the policy in you implementation base on the env
  # # the policy must be a vector in 1-d, 
  # but you can split it to be differet parameters e.g first half to be s,second half to be S, etc
  '''
  if policy is None :
    return np.ones((env.num_stages-1))
 
  unc_actions = policy
  # constraints
  
  inv_const = np.hstack([env.I[env.period, 1:], np.Inf])
  actions = np.minimum(env.c, np.minimum(unc_actions, inv_const))
  return actions
