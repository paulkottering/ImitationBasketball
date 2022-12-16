import numpy as np

def rescale(position, xory, attacking_right):
    if xory == 'x':
        if attacking_right:
            return (100 - position) / 100
        else:
            return position / 100

    elif xory == 'y':
        return position / 50


def makestateaction1(moments, i, attacking_team_first, attacking_right):
    current = moments[i]
    if attacking_team_first == True:
        O = 0
        D = 5
    else:
        O = 5
        D = 0

    state = []
    for l in range(5):
        state.append(rescale(moments[i].players[O + l].x, 'x', attacking_right))
        state.append(rescale(moments[i].players[O + l].y, 'y', attacking_right))

    for l in range(5):
        state.append(rescale(moments[i].players[D + l].x, 'x', attacking_right))
        state.append(rescale(moments[i].players[D + l].y, 'y', attacking_right))

    state.append(rescale(moments[i].ball.x, 'x', attacking_right))
    state.append(rescale(moments[i].ball.y, 'y', attacking_right))

    action = []
    for k in range(5):
        action.append(rescale(moments[i + 1].players[D + k].x, 'x', attacking_right)
                      - rescale(moments[i].players[D + k].x, 'x', attacking_right))
        action.append(rescale(moments[i + 1].players[D + k].y, 'y', attacking_right)
                      - rescale(moments[i].players[D + k].y, 'y', attacking_right))

    reward = np.sqrt((rescale(moments[i].ball.x, 'x', attacking_right) - 0.5) ** 2
                     + (rescale(moments[i].ball.y, 'y', attacking_right)) ** 2)

    state = np.clip(state, 0, 1)
    action = np.clip(action, -0.2, 0.2)

    if i == 0:
        start = True
    else:
        start = False

    return state, action, reward, start


def makestateaction2(moments, i, attacking_team_first, attacking_right):
    if attacking_team_first == True:
        O = 0
        D = 5
    else:
        O = 5
        D = 0

    state = []
    for l in range(5):
        state.append(rescale(moments[i].players[O + l].x, 'x', attacking_right))
        state.append(rescale(moments[i].players[O + l].y, 'y', attacking_right))

    for m in range(5):
        state.append(rescale(moments[i + 1].players[O + m].x, 'x', attacking_right)
                     - rescale(moments[i].players[O + m].x, 'x', attacking_right))
        state.append(rescale(moments[i + 1].players[O + m].y, 'y', attacking_right)
                     - rescale(moments[i].players[O + m].y, 'y', attacking_right))

    for l in range(5):
        state.append(rescale(moments[i].players[D + l].x, 'x', attacking_right))
        state.append(rescale(moments[i].players[D + l].y, 'y', attacking_right))

    state.append(rescale(moments[i].ball.x, 'x', attacking_right))
    state.append(rescale(moments[i].ball.y, 'y', attacking_right))

    action = []
    for k in range(5):
        action.append(rescale(moments[i + 1].players[D + k].x, 'x', attacking_right)
                      - rescale(moments[i].players[D + k].x, 'x', attacking_right))
        action.append(rescale(moments[i + 1].players[D + k].y, 'y', attacking_right)
                      - rescale(moments[i].players[D + k].y, 'y', attacking_right))

    reward = np.sqrt((rescale(moments[i].ball.x, 'x', attacking_right) - 0.5) ** 2
                     + (rescale(moments[i].ball.y, 'y', attacking_right)) ** 2)


    state[:10] = np.clip(state[:10], 0, 1)
    state[10:20] = np.clip(state[10:20], -0.2, 0.2)
    state[20:32] = np.clip(state[20:32], 0, 1)
    action = np.clip(action, -0.2, 0.2)

    if i == 0:
        start = True
    else:
        start = False

    return state, action, reward, start


def makestateaction3(moments, i, attacking_team_first, attacking_right):
    if attacking_team_first == True:
        O = 0
        D = 5
    else:
        O = 5
        D = 0

    state = []
    action = []

    for l in range(5):
        state.append(rescale(moments[i].players[O + l].x, 'x', attacking_right))
        state.append(rescale(moments[i].players[O + l].y, 'y', attacking_right))

    state.append(rescale(moments[i].ball.x, 'x', attacking_right))
    state.append(rescale(moments[i].ball.y, 'y', attacking_right))

    for l in range(5):
        action.append(rescale(moments[i].players[D + l].x, 'x', attacking_right))
        action.append(rescale(moments[i].players[D + l].y, 'y', attacking_right))

    reward = np.sqrt((rescale(moments[i].ball.x, 'x', attacking_right) - 0.5) ** 2
                     + (rescale(moments[i].ball.y, 'y', attacking_right)) ** 2)

    state = np.clip(state, 0, 1)
    action = np.clip(action, 0, 1)

    if i == 0:
        start = True
    else:
        start = False

    return state, action, reward, start
