from typing import Callable, Tuple
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from model import Model, Actions
np.random.seed(2024)


def q_learning(
    model: Model,
    maxit: int = 100,
    maxit_episode: int = 20,
    epsilon: float = 0.1,
    alpha: float = 1.,
) -> Tuple[NDArray, NDArray]:
    """
    Q-learning
        maxit: int, max iteration of SARSA, i.e. number of episodes
        maxit_episode: int, max movements in each episode
        epsilon: float, the exploration parameter
        alpha: float, the learning rate
    """
    # Q = np.random.randn(model.num_states, len(Actions))
    Q = np.zeros((model.num_states, len(Actions)))
    Q[-1, :] = 0
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))

    for _ in tqdm(range(maxit)):
        s = model.states[np.random.randint(0, model.num_states - 1)]
        for _ in range(maxit_episode):
            if s is model.goal_state:
                break                
            coin = np.random.choice([0, 1], size=1, p=[1 - epsilon, epsilon])
            a_idx = np.random.randint(0, len(Actions)) if coin else np.argmax(Q[s, :])
            a = Actions(a_idx)
            r = model.reward(s, a)
            s_ = model.cell2state(model._result_action(model.state2cell(s), a))
            Q[s, a_idx] = Q[s, a_idx] + alpha * (r + model.gamma * np.max(Q[s_, :]) - Q[s, a_idx])
            s = s_
    
    for s in model.states:
        pi[s] = Actions(np.argmax(Q[s, :]))
        V[s] = np.max(Q[s, :])
    return V, pi