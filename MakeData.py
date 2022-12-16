import argparse
import os
import pandas as pd
import numpy as np
import math

from Utils.MakeStateAction import makestateaction1, makestateaction2, makestateaction3

def parse_args():
    """
    Helper function to set up command line argument parsing
    """
    parser = argparse.ArgumentParser()
    # Set up arguments
    parser.add_argument( "-j","--jsonfilesdirectory",default="JsonFiles",type=str,
                        help = 'which json directory to grab')
    parser.add_argument("-v", "--version", default="v1", type=str, choices=["v1","v2","v3"],
                        help = 'which version to perform')
    parser.add_argument("-n", "--name", default="Tester", type=str,
                        help = 'what name to save the npz file under')
    parser.add_argument("-fl", "--filelimit", default=0, type=int,
                        help = 'limit the number of json files to use in the directory')
    parser.add_argument("-i", "--intervaldistance", default=15, type=int,
                        help = 'how many frames per step in data')
    return parser.parse_args()

class Moment:
    def __init__(self, moment):
        self.quarter = moment[0]
        self.game_clock = moment[2]
        self.shot_clock = moment[3]
        ball = moment[5][0]
        self.ball = Ball(ball)
        players = moment[5][1:]
        self.players = [Player(player) for player in players]


class Player:
    def __init__(self, player):
        self.id = player[1]
        self.x = player[2]
        self.y = player[3]


class Ball:
    def __init__(self, ball):
        self.x = ball[2]
        self.y = ball[3]
        self.z = ball[4]


def start(version, path_to_json, interval_distance, actions, observations, rewards, episode_starts):
    data_frame = pd.read_json(path_to_json)
    last_default_index = len(data_frame) - 1

    for event_index in range(last_default_index):

        eventnum = data_frame['events'][event_index]

        moments = []

        number_of_moments = len(eventnum['moments'])

        for i in range(math.floor(number_of_moments / interval_distance)):

            moment = eventnum['moments'][i * interval_distance]

            moment_formatted = Moment(moment)
            if len(moment_formatted.players) < 10:
                break
            else:
                moments.append(moment_formatted)

        if len(moments) >= 5:
            actions, observations, rewards, episode_starts = save(version,moments, actions,
                                                                  observations, rewards, episode_starts)

    return actions, observations, rewards, episode_starts


def save(version, moments, actions, observations, rewards, episode_starts):
    teams_sum = 0
    side_sum = 0
    for momentindex in range(len(moments) - 1):
        player_to_ball_distance = [np.sqrt((moments[momentindex].players[j].x - moments[momentindex].ball.x) ** 2
                                           + (moments[momentindex].players[j].y - moments[momentindex].ball.y) ** 2)
                                   for j in range(10)]

        if np.argmin(player_to_ball_distance) > 4:
            teams_sum += 1
        if moments[momentindex].ball.x > 50:
            side_sum += 1

    attacking_team_first = 0
    attacking_right = 0

    if teams_sum / (len(moments) - 1) < 0.1:
        attacking_team_first = True

    elif teams_sum / (len(moments) - 1) > 0.9:
        attacking_team_first = False

    if side_sum / (len(moments) - 1) < 0.2:
        attacking_right = True

    elif side_sum / (len(moments) - 1) > 0.8:
        attacking_right = False

    if type(attacking_team_first) is bool and type(attacking_right) is bool:

        for momentindex in range(len(moments) - 1):
            if version == "v1":
                state, action, reward, done = makestateaction1(moments, momentindex, attacking_team_first,
                                                               attacking_right)
            elif version == "v2":
                state, action, reward, done = makestateaction2(moments, momentindex, attacking_team_first,
                                                              attacking_right)
            elif version == "v3":
                state, action, reward, done = makestateaction3(moments, momentindex, attacking_team_first,
                                                              attacking_right)

            observations.append(state)
            actions.append(action)
            rewards.append(reward)
            episode_starts.append(done)

    return actions, observations, rewards, episode_starts


def main(**kwargs):

    json_list = os.listdir(kwargs.get("jsonfilesdirectory"))[1:]

    interval_distance = kwargs.get("intervaldistance")

    actions = []
    observations = []
    rewards = []
    episode_starts = []

    if kwargs.get("filelimit") != 0:
        num_files = kwargs.get("filelimit")
    else:
        num_files = len(json_list)
    print("Starting Data Refactoring ...")
    for i in range(num_files):
        print('File ',i)
        json = json_list[i]
        json_path = kwargs.get("jsonfilesdirectory") + "/" + json
        actions, observations, rewards, episode_starts = start(kwargs.get("version"), json_path, interval_distance,
                                                               actions, observations, rewards, episode_starts)

    episode_returns = np.zeros((len(observations) - 1,))

    if kwargs.get("version") == "v1":
        state_size = 22
    elif kwargs.get("version") == "v2":
        state_size = 32
    elif kwargs.get("version") == "v3":
        state_size = 12

    observations = np.concatenate(observations).reshape((-1,) + (state_size,))
    actions = np.concatenate(actions).reshape((-1,) + (10,))

    episode_starts = np.array(episode_starts[:-1])
    rewards = np.array(rewards)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    print('Completed Data Refactoring. Saving File.')

    save_path = 'Demos/' + kwargs.get("name") + kwargs.get("version")
    np.savez(save_path, **numpy_dict)



if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)