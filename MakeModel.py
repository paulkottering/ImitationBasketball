import argparse
from stable_baselines.gail import ExpertDataset
from stable_baselines import GAIL
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
from torch import nn

from Utils.FullyConnected import FullConnec
from Utils.BballEnvs import BballScape1,BballScape2,BballScape3

def parse_args():
    """
    Helper function to set up command line argument parsing
    """
    parser = argparse.ArgumentParser()
    # Set up arguments
    parser.add_argument("-d","--npzfile",type=str,
                        help = 'which npz file to grab')
    parser.add_argument("-v", "--version", default="v1", type=str, choices=["v1","v2","v3"],
                        help = 'which version to perform')
    parser.add_argument("-n", "--name", default="TesterModel", type=str,
                        help = 'what name to save model as')
    parser.add_argument("-t", "--totaltimesteps", default=1000000 ,type=int,
                        help = 'number of timesteps to perform')
    parser.add_argument("-g", "--GAIL", default='t' ,type=str,
                        help = 'Whether to use GAIL')
    return parser.parse_args()


def main(**kwargs):

    if kwargs.get("GAIL") == 't':
        print('Loading Dataset',kwargs.get("npzfile"))
        dataset = ExpertDataset(expert_path=kwargs.get("npzfile"), verbose=1)

        print('Creating Environment', kwargs.get("version"))
        if kwargs.get("version") == "v1":
            env = BballScape1()
        elif kwargs.get("version") == "v2":
            env = BballScape2()
        elif kwargs.get("version") == "v3":
            env = BballScape3()

        print('Creating Model')
        model = GAIL('MlpPolicy', env, dataset, verbose=1)

        print('Learning ... ')
        model.learn(total_timesteps=kwargs.get("totaltimesteps"))

        print('Saving Model ... ')
        model.save("Models/" + kwargs.get("name") + kwargs.get("version"))
    else:
        if kwargs.get("version") != "v3":
            raise Exception("Non GAIL can only be used in version 3")
        data = np.load(kwargs.get("npzfile"))
        x_s =  data['obs']
        y_s =  data['actions']

        h = 1
        while not data['episode_starts'][h]:
            h += 1

        train_x = torch.tensor(x_s[h:])
        train_y = torch.tensor(y_s[h:])

        funcdata = TensorDataset(train_x,train_y)
        trainloader = DataLoader(funcdata,batch_size=64,shuffle=True)

        model = FullConnec()
        loss_function = nn.MSELoss()
        optimizer= torch.optim.SGD(model.parameters(), lr=0.001)
        epochs = 50

        for epoch in range(epochs):
            for inputs,targets in trainloader:
                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = loss_function(outputs.float(),targets.float())
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(),"Models/" + kwargs.get("name") + kwargs.get("version"))



if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)