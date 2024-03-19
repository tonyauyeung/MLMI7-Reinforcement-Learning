from typing import Tuple
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from model import Model, Actions


def sarsa(
    model: Model,
    maxit: int = 100,
    n_episode: int = 500,
    epsilon: float = 0.1,
    alpha: float = 1.,
) -> Tuple[NDArray]:
    """
    State-Action-Reward-Action-State, SARSA
        maxit: int, max iteration of SARSA, i.e. number of episodes
        maxit_episode: int, max movements in each episode
        epsilon: float, the exploration parameter
        alpha: float, the learning rate
    """
    Q = np.zeros((model.num_states, len(Actions)))
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))
    rewards = np.zeros((n_episode, ))
    # history = [np.copy(Q)]
    iters = np.zeros((n_episode, ))
    for i in range(n_episode):
        s = model.start_state
        coin = np.random.choice([0, 1], size=1, p=[1 - epsilon, epsilon])
        a = np.random.randint(0, len(Actions)) if coin else np.argmax(Q[s, :])
        for _ in range(maxit):
            r = model.reward(s, a)
            rewards[i] += r
            s_ = model.next_state(s, a)
            coin = np.random.choice([0, 1], size=1, p=[1 - epsilon, epsilon])
            a_ = np.random.randint(0, len(Actions)) if coin else np.argmax(Q[s_, :])
            Q[s, a] = Q[s, a] + alpha * (r + model.gamma * Q[s_, a_] - Q[s, a])
            s, a = s_, a_
            iters[i] += 1
            if s == model.goal_state:
                break                
        # history.append(np.copy(Q))
    
    pi = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    return V, pi, rewards, iters


def expected_sarsa(
    model: Model,
    maxit: int = 100,
    n_episode: int = 500,
    epsilon: float = 0.1,
    alpha: float = 1.,
) -> Tuple[NDArray]:
    
    Q = np.zeros((model.num_states, len(Actions)))
    Q[-1, :] = 0
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))
    rewards = np.zeros((n_episode, ))
    iters = np.zeros((n_episode, ))
    for i in range(n_episode):
        s = model.start_state
        coin = np.random.choice([0, 1], size=1, p=[1 - epsilon, epsilon])
        a = np.random.randint(0, len(Actions)) if coin else np.argmax(Q[s, :])
        for _ in range(maxit):       
            r = model.reward(s, a)
            rewards[i] += r
            s_ = model.next_state(s, a)
            coin = np.random.choice([0, 1], size=1, p=[1 - epsilon, epsilon])    
            a_ = np.random.randint(0, len(Actions)) if coin else np.argmax(Q[s_, :])
            a_s = np.where(Q[s_, :] == np.max(Q[s_, :]))[0]
            expected_q = 0.
            for j in range(4):
                expected_q += ((1 - epsilon) + epsilon / len(Actions)) * Q[s_, j] if j in a_s else epsilon / len(Actions) * Q[s_, j]

            Q[s, a] = Q[s, a] + alpha * (r + model.gamma * expected_q - Q[s, a])
            
            s, a = s_, a_
            iters[i] += 1
            if s == model.goal_state:
                break         
    
    pi = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    return V, pi, rewards, iters
