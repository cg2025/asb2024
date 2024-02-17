## Command line execution example
python -m CodeColab.UE.ElbowFixTarget.Scripts.TrainHealthy tgtrnd -1 -1 100000

## Four arguments are required
1. 'tgtrnd' : Filename identifier for logs and policies files (Saved in logs and Policies folder)
2. weight_value : set to -1 if it is to be selected at random at runtime o.w. specify the value
3. target_value : set to -1 if target angle is to be selected at random o.w. specify the value
4. train_steps : specify number of training steps

## Command line execution example
python -m CodeColab.UE.ElbowFixTarget.Scripts.TrainExo tgtrnd -1 -1 100000

Same argument set as above
Exo control parameters are specified inside the code as a list of values (bottom of the code)


