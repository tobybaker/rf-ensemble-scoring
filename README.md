# Random Forest Structural Ensemble Scoring

A novel method to assess the convergence of protein structural ensembles using random forests. The structures in the ensemble are given one of N sequential labels. A random forest model is trained to predict the correct label for a given structure. The convergence score is the average probability the model assigns previously unseen structures to their correct label. As convergence is reached and states revisited, similar structures will be given different labels and thus the average probability will decrease.

_caconversion.py_ provides the necessary code to convert an ensemble into a set of pairwise Alpha-Carbon distances suitable for scoring. 

This is passed to _forestscore.py_ along with other parameters and returns the probability-based score.
