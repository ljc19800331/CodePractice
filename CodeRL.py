
'''
Some random ideas:
1. inference from the GP map -- high uncertainty region measured by the probability --
    a. p(high|MAP) = ?
    b. p(low|MAP) = ? -- transition probability
        if choose a higher p(low|MAP)
2. what is the learned parameters for a model-based RL?
    a. weight parameters for each constrain of the level ...
    b. The weights for each step (The importance between each steps from the previous history)
    c. continuous reward -- long-time reward -- this reward learns the feature from the studies
    d. what is the learnable model? -- maybe it is similar to a Gaussian-Process Regression Problems -- pending
3. what is the objective function for this problem?
    a. learn the state transition function based on several constraints

'''