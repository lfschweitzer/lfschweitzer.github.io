import numpy as np
import sys

NUM_CASES  = 1000
PREVALENCE = 0.1
NOISE      = 2

cases  = 1*(np.random.rand(NUM_CASES) < PREVALENCE)
scores = np.exp(cases + NOISE*(np.random.rand(NUM_CASES))) / np.exp(NOISE+1)

def compute_cost(scores, cases, t):
    
    cost_a = 1
    cost_b = 3
    
    scenario_a = np.sum((scores >= t) == cases)
    scenario_b = np.sum((scores >= t) != cases)

    cost = scenario_a * cost_a + scenario_b * cost_b
    
    return cost

min = np.inf
bestT = 0

for t in range(0, 101):
    t = t/100.0
    
    cost = compute_cost(scores, cases, t)
    if cost < min:
        min = cost
        bestT = t
    
    
print(compute_cost(scores, cases, t))
# print(computeCost2(scores, cases, t))
print(bestT)