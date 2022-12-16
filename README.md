# ImitationBasketball

The three different environments are described here:

| Version                 | v1                                                     | v2                                                                                                                         | v3                                                    |
|-------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| State Space Descrption  | spaces.Box(shape=(22,))                                | spaces.Box(shape=(32,))                                                                                                    | spaces.Box(shape=(12,))                               |
| State Space Dimension   | X and Y co-ordinates of all 10 players and the ball    | X and Y co-ordinates of all 10 players and the ball. As well as the X and Y co-ordinate changes for the offensive players. | X and Y co-ordinates of 5 offensive players and Ball. |
| Action Space Descrption | spaces.Box(shape=(10,))                                | spaces.Box(shape=(10,))                                                                                                    | spaces.Box(shape=(10,))                               |                      |
| Action Space Dimension  | X and Y co-ordinate changes of all 5 defensive players |  X and Y co-ordinate changes of all 5 defensive players                                                                     | X and Y co-ordinates of 5 defensive players.          |                         |                         |

When we collect data we ensure that the X and Y co-ordinates are rescale to be betwen 0 and 1, as such the state space for X and Y co-ordinates are limited to be between 0 and 1.


'requirements.txt' specifies what packages are need to run the algorithms. 
'MakeData.py' converts a directory of json files containing the raw SportVU data into a '.npz' file and saves this in the Demos file.
'MakeModel.py' retrieves data from an npz file and builds and trains a model. It then saves the model in the Models file according to the name given. 
'TestModel.py' takes a model and tests it. It does this by showing how the model would have responded to a given offensive trajectory.
The relevant OpenAI Gym Environments are in the  in Utils 'BballEnv', 'BballEnv-v2' and 'BballEnv-v3'.


To run a quick experiment using the limited number of data that has been uploaded to github in the JsonFiles folder, to see how the repository works, thry the following sequence:

First set up a virtual environment and install the necessary packages:

``` 
pip install requirements.txt
```

Then, clean the data and save it in a usable format:

```
python MakeData.py -j JsonFiles -v v1 -n READMETest -fl 2
```
Then, use the data to build a model:
```
python MakeModel.py -d Demos/READMETestv1.npz -v v1 -n READMETest -t 2000
```
Then, use the model to attempt to predict a trajectory:
``` 
python TestModel.py -m Models/READMETestv1.zip -v v1 -n READMETest -d Demos/READMETestv1.npz
```

Below we can see some of the plots from the different version testing on new trajectories:

![Alt text](TestImages/FullDatav1.png?raw=true "Title")

![Alt text](TestImages/FullDatav2.png?raw=true "Title")

![Alt text](TestImages/FullDatav3.png?raw=true "Title")
