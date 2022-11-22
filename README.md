# ImitationBasketball


Notebook makes pickle files with state-action trajectories. Not completed yet, need to clean. 

I will be making a custom environment in OpenAI gym (https://blog.paperspace.com/creating-custom-environments-openai-gym/)

This requires simply a class with the init, step, rest and render functions. 

Then it must be registered with the open ai gym https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym.

Then we can run the GAIL simply using the structure from the ilypt readme. We need to have saved the demos for the 'expert-demos' file. 

To do GAIL I will adapt: https://github.com/mitre/ilpyt
