# Predict Bike Sharing Demand with AutoGluon (Kaggle competition)

## Exploratory data analysis and feature creation

### Key findings:
- The datetime column wasn't informative as is, so I split it into month, dayofweek and hour columns.  
I did it because I believe the demand can better be predicted better based on month (season), day of week (workday or weekend)
and hour (rush hour).
- There are very few entries on holydays, so bootstrapping may help (I didn't try it).
- There may be some outliers in the count column, but I didn't remove them, as I am not sure if they are errors or not.

## Hyperparameters tuning

The model didn't performe much better after hyperparameters tuning. AutoGluon already does a good job with the default parameters.

### Future work:
I think there's more to be done in feature engineering, for example removing outliers, binning, normalizing, etc.
Maybe extract more features from the datetime column, and split categorical columns using one-hot encoding.
I wonder if bootstrapping would make a difference.  
(AutoGluon and some of the models probably take care of most of these, but I would like to try it anyway)

### Table of hyperparameters and Kaggle score:
There are more hyperparameters that I tried, but I didn't include them in the table.

| model        | GBM                                                                          | CAT                                                                 | XGB                                                       | score   |
|--------------|------------------------------------------------------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------|---------|
| initial      | default                                                                      | default                                                             | default                                                   | 1.8043  |
| add_features | default                                                                      | default                                                             | default                                                   | 0.47456 |
| hpo1         | num_leaves=40, min_child_samples=20                                          | depth=6, l2_leaf_reg=3, learning_rate=0.1, bagging_temperature=0.75 | max_depth=6, subsample=.75, colsample_bytree=.75          | 0.65364 |
| hpo2         | num_leaves=31, learning_rate=0.1, min_child_samples=10, bagging_fraction=0.8 | depth=8, l2_leaf_reg=3, learning_rate=0.1, bagging_temperature=1    | eta=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8 | 0.52443 |
| hpo3         | [(extra_trees=True, ag_args={'name_suffix': 'XT'}), default, 'GBMLarge']     | default                                                             | -                                                         | 0.48337 |
| hpo4         | [(extra_trees=True, ag_args={'name_suffix': 'XT'}), default, 'GBMLarge']     | default                                                             | -                                                         | 0.47159 |

I tried to prevent overfitting by reducing depth etc., but it didn't help - probably because of the ensemble and random nature of the models.
(used hyperparameter_tune_kwargs at hpo4)

## Summary

AutoGluon provides a very easy and powerful way to train and predict with many models.  
With a little bit of feature engineering, I was able to predict the bike sharing demand with a good score.