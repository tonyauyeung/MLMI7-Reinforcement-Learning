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
            s = s_
            a = a_
            if s == model.goal_state:
                break                
    
    pi = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    return V, pi, rewards


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

    for i in range(n_episode):
        # s = model.states[np.random.randint(0, model.num_states - 1)]
        s = model.start_state
        coin = np.random.choice([0, 1], size=1, p=[1 - epsilon, epsilon])
        a_idx = np.random.randint(0, len(Actions)) if coin else np.argmax(Q[s, :])
        a = Actions(a_idx)
        for _ in range(maxit):       
            r = model.reward(s, a)
            rewards[i] += r
            s_ = model.next_state(s, a)
            coin = np.random.choice([0, 1], size=1, p=[1 - epsilon, epsilon])
            
            a_idx_ = np.random.randint(0, len(Actions)) if coin else np.argmax(Q[s_, :])
            a_idx_s = np.where(Q[s_, :] == np.max(Q[s_, :]))[0]
            expected_q = 0.
            for j in range(4):
                expected_q += ((1 - epsilon) + epsilon / len(Actions)) * Q[s_, j] if j in a_idx_s else epsilon / len(Actions) * Q[s_, j]

            Q[s, a_idx] = Q[s, a_idx] + alpha * (r + model.gamma * expected_q - Q[s, a_idx])
            
            s, a, a_idx = s_, Actions(a_idx_), a_idx_
            if s == model.goal_state:
                break         
    
    pi = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    return V, pi, rewards


if __name__ == '__main__':
    from world_config import cliff_world
    import matplotlib.pyplot as plt
    model = Model(cliff_world)
    epsilon = 0.1
    alpha = .5
    n_episode = 500
    n_reps = 100
    cum_r_sarsa = np.zeros(n_episode)
    # cum_r_expected_sarsa = np.zeros(n_episode)
    np.random.seed(42)
    for _ in tqdm(range(n_reps)):
        V_sarsa, pi_sarsa, rewards_sarsa = sarsa(model, epsilon=epsilon, alpha=alpha, n_episode=n_episode)
        # V_expected_sarsa, pi_expected_sarsa, rewards_expected_sarsa = sarsa(model, epsilon=epsilon, alpha=alpha, n_episode=n_episode)
        cum_r_sarsa += rewards_sarsa / n_reps
        # cum_r_expected_sarsa += np.array(rewards_expected_sarsa)
        # cum_r_expected_sarsa /= n_reps
    plt.figure(figsize=(8, 4))
    episodes = np.arange(1, n_episode + 1)
    plt.plot(cum_r_sarsa)
    # plt.plot(episodes, rewards_expected_sarsa)
    plt.ylim([-100, 0])
    plt.show()