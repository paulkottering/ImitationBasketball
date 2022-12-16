import argparse
import matplotlib.pyplot as plt
from stable_baselines import GAIL
import os
import numpy as np
from Utils.BballEnvs import BballScape1,BballScape2,BballScape3
from Utils.FullyConnected import FullConnec
import torch

def parse_args():
    """
    Helper function to set up command line argument parsing
    """
    parser = argparse.ArgumentParser()
    # Set up arguments
    parser.add_argument("-m","--modelfile",type=str,
                        help='which model to grab')
    parser.add_argument("-v", "--version", default="v1", type=str, choices=["v1","v2","v3"],
                        help='which version to perform')
    parser.add_argument("-n", "--name", default="TesterModel", type=str,
                        help='under what name to save images')
    parser.add_argument("-d", "--data", default="TesterModel", type=str,
                        help='path to dataset to use for testing ')
    parser.add_argument("-g", "--GAIL", default='t' ,type=str,
                        help = 'Whether to use GAIL')
    return parser.parse_args()


def main(**kwargs):

    if kwargs.get("GAIL") == 't':
        model = GAIL.load(kwargs.get("modelfile"))
        if kwargs.get("version") == "v1":
            env = BballScape1()
        elif kwargs.get("version") == "v2":
            env = BballScape2()
        elif kwargs.get("version") == "v3":
            env = BballScape3()
    else:
        if kwargs.get("version") != "v3":
            raise Exception("Non GAIL can only be used in version 3")
        model = FullConnec()
        model.load_state_dict(torch.load(kwargs.get("modelfile")))
        model.eval()

    data = np.load(kwargs.get("data"))

    h = 1
    while not data['episode_starts'][h]:
        h += 1

    state_traj = data['obs'][:h - 1]
    action_traj = data['actions'][:h - 1]

    state_model = state_traj[0]

    fig,ax = plt.subplots(figsize=(9, 5))
    img = plt.imread("court.png")
    ax.imshow(img,extent=[0,1,0,1], aspect='auto')

    for index in range(h-1):

        state_now = state_traj[index]
        action_expert = action_traj[index]
        if kwargs.get("GAIL") == 't':
            action_model, _ = model.predict(state_model)
        else:
            action_model = model(torch.tensor(state_model).float())

        if kwargs.get("version") == "v1":
            for i in range(5):
                ax.scatter(state_now[2 * i], state_now[2 * i + 1], c='b')
                ax.scatter(state_now[10 + 2 * i], state_now[10 + 2 * i + 1], c='y')
                x = np.clip(state_model[10 + 2 * i] + action_model[2 * i],0,1)
                y = np.clip(state_model[10 + 2 * i + 1] + action_model[2 * i + 1],0,1)

                ax.scatter(x, y, c='r')
                ax.arrow(state_model[10 + 2 * i],
                          state_model[10 + 2 * i + 1],
                          x-state_model[10 + 2 * i],
                          y-state_model[10 + 2 * i + 1])

            if index != h-2:
                next_state = state_traj[index+1]
                state_model[:9] = next_state[:9]
                state_model[10:20] = np.clip(state_model[10:20] + action_model,0,1)
            plt.title("Trajectory for v1 - 1M Timesteps Trained - 569 Games")

        elif kwargs.get("version") == "v2":
            for i in range(5):
                ax.scatter(state_now[2 * i], state_now[2 * i + 1], c='b')
                ax.scatter(state_now[20 + 2 * i], state_now[20 + 2 * i + 1], c='y')

                x = np.clip(state_model[20 + 2 * i] + action_model[2 * i], 0, 1)
                y = np.clip(state_model[20 + 2 * i + 1] + action_model[2 * i + 1], 0, 1)

                ax.scatter(x, y, c='r')

                ax.arrow(state_model[20 + 2 * i] ,
                          state_model[20 + 2 * i + 1] ,
                          x-state_model[20 + 2 * i],
                          y-state_model[20 + 2 * i + 1])

            if index != h-2:
                next_state = state_traj[index+1]
                state_model[:19] = next_state[:19]
                state_model[20:30] = np.clip(state_model[20:30] + action_model,0,1)
            plt.title("Trajectory for v2 - 1M Timesteps Trained - 569 Games")

        elif kwargs.get("version") == "v3":
            for i in range(5):
                ax.scatter(state_now[2 * i], state_now[2 * i + 1], c='b')
                ax.scatter(action_expert[2 * i], action_expert[2 * i + 1], c='y')
                x = np.clip(action_model[2 * i].detach().numpy(),0,1)
                y = np.clip(action_model[2 * i + 1].detach().numpy(),0,1)
                ax.scatter(x,y, c='r')

            if index != h-2:
                state_model = state_traj[index+1]
            plt.title("Trajectory for v3 - 1M Timesteps Trained- 569 Games")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.scatter(0,0.5,  c ='b', label = 'Attacking Team True')
    ax.scatter(0, 0.5, c='y', label='Defending Team True')
    ax.scatter(0, 0.5, c='r', label='Defending Team Learned')
    ax.scatter(0, 0.5, c='k', label='Basketball Hoop',s=40)
    ax.legend()

    plt.savefig('TestImages/' + kwargs.get("name") + kwargs.get("version"))


if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)