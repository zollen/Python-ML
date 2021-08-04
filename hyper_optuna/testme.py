'''
Created on Aug. 4, 2021

@author: zollen
@desc the best or most used kaggle hyperparameters tuning
@url: https://towardsdatascience.com/why-is-everyone-at-kaggle-obsessed-with-optuna-for-hyperparameter-tuning-7608fdca337c
'''

import optuna

def myobjective(trial):
    x = trial.suggest_float("x", -7, 7)
    y = trial.suggest_float("y", -7, 7)
    return (x - 1) ** 2 + (y + 3) ** 2


'''
direction="maximize"
direction="minimize"
'''
study = optuna.create_study(direction="maximize")
study.optimize(myobjective, n_trials=100)
print(study.best_params)

'''
This is a distinct advantage over other similar tools because after the search is done, 
they completely forget the history of previous trials. Optuna does not!
'''

study.optimize(myobjective, n_trials=100)
print(study.best_params)