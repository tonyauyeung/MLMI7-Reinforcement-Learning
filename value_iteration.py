from typing import Callable, Tuple
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from model import Model, Actions

def value_iteration(
    model: Model, 
    maxit: int = 100, 
    threshold: float = 0.01
) -> Tuple[NDArray, NDArray]:
    """
    Synchronise Value Iteration, SyncVI
        maxit: int, max iteration of VI
        threshold: flaot, control the convergence status of VI
    """
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))

    def compute_value(s, a, reward: Callable):
        return np.sum(
            [
                model.transition_probability(s, s_, a)
                * (reward(s, a) + model.gamma * V[s_])
                for s_ in model.states
            ]
        )
    
    for _ in tqdm(range(maxit)):
        V_new = np.zeros_like(V)
        for s in model.states:
            values = []
            for a in Actions:
                R = model.reward(s, a)
                values.append(compute_value(s, a, lambda *_: R))
            V_new[s] = max(values)
        delta = np.max(np.abs(V_new - V))
        V = np.copy(V_new)
        if delta <= threshold:
            break
    for s in model.states:
        action_index = np.argmax(
            [compute_value(s, a, model.reward) for a in Actions]
        )
        pi[s] = Actions(action_index)
    return V, pi


def value_iteration_async(
    model: Model, 
    maxit: int = 100, 
    threshold: float = 0.01
) -> Tuple[NDArray, NDArray]:
    """
    Asynchronise Value Iteration, ASyncVI
        maxit: int, max iteration of VI
        threshold: flaot, control the convergence status of VI
    """
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))

    def compute_value(s, a, reward: Callable):
        return np.sum(
            [
                model.transition_probability(s, s_, a)
                * (reward(s, a) + model.gamma * V[s_])
                for s_ in model.states
            ]
        )
    
    for _ in tqdm(range(maxit)):
        delta = -1e+6
        for s in model.states:
            values = []
            for a in Actions:
                R = model.reward(s, a)
                values.append(compute_value(s, a, lambda *_: R))
            delta = max(delta, np.abs(V[s] - max(values)))
            # update value function without storing the previous value
            V[s] = max(values)
        if delta <= threshold:
            break
    for s in model.states:
        action_index = np.argmax(
            [compute_value(s, a, model.reward) for a in Actions]
        )
        pi[s] = Actions(action_index)
    return V, pi
