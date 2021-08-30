# Implementation
1. CaseCollection counts the number of non-redundant positive and negative cases.
1. Bar plot of fraction of positive cases for each class.
1. CaseCollection - serialize/deserialize
2. CaseMultiCollection - serialize/deserialize
3. Refactor codes for CaseMultiCollection
4. Tests for CaseMultiCollection
5. Eliminate CaseQuery
6. Jupyter Notebook updates
7. Prune cases that give incorrect classifications for training data. May
   want to do this using cross validation. Create a new module (CasePruner)
   that employs several kinds of pruning. Need a criteria for the class for
   when a Case is pruned (e.g., fraction of samples that it applies to).
   
# Analysis
1. Analysis of false positives for classes.
