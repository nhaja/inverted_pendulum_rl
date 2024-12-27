from util import StateEnum as X
from util import normalize_angle
import numpy as np

def reward_simple(state, action, step) -> float:
    state = state.flatten()
    action = action.flatten()
    return float(
        -(0.1 * state[0] ** 2 + 0.01 * state[1] ** 2 + 2.0 * step * normalize_angle(state[2]) ** 2 +  0.01 * state[3] ** 2 + 1000 * action ** 2)
    )

def reward_uptime(state, action, step) -> float:
    state = state.flatten()
    action = action.flatten()
    return 1.0 - normalize_angle(state[2]) ** 2 - 0.01 * np.abs(state[1]) - 0.01 * np.abs(state[3]) - np.abs(state[0]) - np.abs(action[0])


def reward_double(state, action, step) -> float:
    state = state.flatten()
    action = action.flatten()
    return 5.0 - normalize_angle(state[2]) ** 2 - normalize_angle(state[3]) ** 2 - 0.01 * np.abs(state[1]) - 0.01 * np.abs(state[4]) - 0.01 * np.abs(state[5]) - np.abs(state[0]) - np.abs(action[0])

def reward_triple(state, action, step) -> float:
    state = state.flatten()
    action = action.flatten()
    return 10.0 - normalize_angle(state[2]) ** 2 - normalize_angle(state[3]) ** 2 - normalize_angle(state[4]) ** 2 - 0.01 * np.abs(state[1]) - 0.01 * np.abs(state[5]) - 0.01 * np.abs(state[6]) - 0.01 * np.abs(state[7]) - 10.0 * np.abs(state[0]) - 10.0 * np.abs(action[0])
