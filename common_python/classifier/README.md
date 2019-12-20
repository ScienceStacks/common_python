# Classifier Directory
This directory contains modules for classification.
Most modules are for meta-classifiers, classifiers built from
other classifiers.
Some modules are for evaluating classifiers.

## Ensembles
- classifier\_collection.py provides methods for a group of classifiers, especially cross validation.
- classifier\_ensemble.py extends a single instance classifier into an ensemble.
- classifier\_ensemble\_random\_ensemble enapsulate a random forest as a ClassifierEnsemble

## Classifier evaluation
- experiment\_hypergrid creates synthetic data for a hypergrid and evaluates classifier accuracy.

## Feature noise
- feature\_noise\_classifier provides various techniques for reducing noise including feature replicas and thresholds.
