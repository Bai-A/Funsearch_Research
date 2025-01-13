"""Finds best policy for the lowest cost.

On every iteration, improve priority_v1 over the priority_vX methods from previous iterations.
Make only small changes.
Try to make the code short.
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
    prob = env.demand_dist.pmf(env.D, **env.dist_param)
    
    # Return negative of expected profit
    return -1 / env.num_periods * np.sum(prob * rewards)
  
  def optimize_inventory_policy(env_name, fun,
    init_policy=None, env_config={}, method='Powell'):
    
    env = or_gym.make(env_name, env_config=env_config)
    
    if init_policy is None:
        init_policy = np.ones(env.num_stages-1)
        
    # Optimize policy
    out = minimize(fun=fun, x0=init_policy, args=env, 
        method=method)
    #if we want the llm to create some new policy instead of using the existing one, we can not set up basic rules like we did before which isfinding a
    #static number as constrain, we need to teach the llm how to find these optimized number.in order to do so, we need to put the more code into llm but in previous experiment
    #it doesnt work
 
    #how do we make sure the policy llm generate is valid or even working? or can we think of a way that doesnt require llm to do such thing?
    policy = out.x.copy()
    
    # Policy must be positive integer
    policy = np.round(np.maximum(policy, 0), 0).astype(int)
    
    return policy, out
  env_name='InvManagement-v1'
  env_config = {}
  policy, out = optimize_inventory_policy('InvManagement-v1',
    dfo_func)
  print("Re-order levels: {}".format(policy))
  print("DFO Info:\n{}".format(out))

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
  This is a re-order up-to policy for you to start. This means that for
  each node in the network, if the inventory at that node 
  falls below the level denoted by the policy, we will 
  re-order inventory to bring it to the policy level.
  
  Design new policy that fits the problem.and improve the score.
  '''
  # Get echelon inventory levels
  if env.period == 0:
    inv_ech = np.cumsum(env.I[env.period] +
      env.T[env.period])
  else:
    inv_ech = np.cumsum(env.I[env.period] +
      env.T[env.period] - env.B[env.period-1, :-1])
  # Get unconstrained actions
  unc_actions = policy - inv_ech
  unc_actions = np.where(unc_actions>0, unc_actions, 0)
  # Ensure that actions can be fulfilled by checking 
  # constraints
  inv_const = np.hstack([env.I[env.period, 1:], np.Inf])
  actions = np.minimum(env.c, np.minimum(unc_actions, inv_const))
  return actions


