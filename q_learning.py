from typing import Callable, Tuple
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from model import Model, Actions


def q_learning(
    model: Model,
    maxit: int = 100,
    n_episode: int = 500,
    epsilon: float = 0.1,
    alpha: float = 1.,
) -> Tuple[NDArray]:
    """
    Q-learning
        maxit: int, max iteration of SARSA, i.e. number of episodes
        maxit_episode: int, max movements in each episode
        epsilon: float, the exploration parameter
        alpha: float, the learning rate
    """
    Q = np.zeros((model.num_states, len(Actions)))
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))
    rewards = np.zeros((n_episode, ))

    for i in range(n_episode):
        s = model.start_state
        for _ in range(maxit):          
            coin = np.random.choice([0, 1], size=1, p=[1 - epsilon, epsilon])
            a = np.random.randint(0, len(Actions)) if coin else np.argmax(Q[s, :])
            r = model.reward(s, a)
            rewards[i] += r
            s_ = model.next_state(s, a)
            Q[s, a] = Q[s, a] + alpha * (r + model.gamma * np.max(Q[s_, :]) - Q[s, a])
            s = s_
            if s == model.goal_state:
                break

    pi = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    return V, pi, rewards

# from world_config import cliff_world
# model = Model(cliff_world)
# # V_sarsa, pi_sarsa, rewards_sarsa = sarsa(model, maxit=500)
# V_q_learning, pi_q_learning, rewards_q_learning = q_learning(model)