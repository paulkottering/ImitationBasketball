# ImitationBasketball


The relevant OpenAI Gym Environments are in the files 'BballEnv', 'BballEnv-v2' and 'BballEnv-v3'.

The three different environments are described here:

| Version                 | v1                                                     | v2                                                                                                                         | v3                                                    |
|-------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| State Space Descrption  | spaces.Box(shape=(22,))                                | spaces.Box(shape=(32,))                                                                                                    | spaces.Box(shape=(12,))                               |
| State Space Dimension   | X and Y co-ordinates of all 10 players and the ball    | X and Y co-ordinates of all 10 players and the ball. As well as the X and Y co-ordinate changes for the offensive players. | X and Y co-ordinates of 5 offensive players and Ball. |
| Action Space Descrption | spaces.Box(shape=(10,))                                | spaces.Box(shape=(10,))                                                                                                    | spaces.Box(shape=(10,))                               |                      |
| Action Space Dimension  | X and Y co-ordinate changes of all 5 defensive players |  X and Y co-ordinate changes of all 5 defensive players                                                                     | X and Y co-ordinates of 5 defensive players.          |                         |                         |

When we collect data we ensure that the X and Y co-ordinates are rescale to be betwen 0 and 1, as such the state space for X and Y co-ordinates are limited to be between 0 and 1.

Here we provide a description of the files and their functionality:

'requirements.txt' specifies what packages are need to run the algorithms. 

'MakeData.py' converts a directory of json files containing the raw SportVU data into a '.npz' file and saves this in the Demos file.
The file takes in three arguments, the path to the directory of json files, the version of the environment data which you would like to save and the name of the data collection.
For example, the following command line would convert it to a v2 data set:

```
python MakeData.py -j JsonFiles -v v2 -n Name
```

'MakeModel.py' retrieves data from an npz file and builds and trains a model. It then saves the model in the Models file according to the name given. 
The argument '-t' indicates the number of timesteps to train the model. The default is 1 million:

```
python MakeModel.py -d Demos/PathToData.npz -v v2 -n Name -t 10000
```

'TestModel.py' takes a model and tests it. It does this by showing how the model would have responded to a given offensive trajectory.
The file takes in the trained model as input, the version, and the path to a trajectory. 
It then saves an image in the Images folder, which displays the trajectories of the offensive and defensive players on the court.  :

```
python TestModel.py -m Models/TestMakev2.zip -v v2 -n Plotting -d Demos/TestMakeDatav2.npz 
```