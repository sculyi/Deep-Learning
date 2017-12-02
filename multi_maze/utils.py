import numpy as np
def get_global_reward( rewards):
    if -10.0 in rewards:
        return -10.0, True
    if np.sum(rewards) == 10 * len(rewards):
        return 1.0,True
    return 0.0, False