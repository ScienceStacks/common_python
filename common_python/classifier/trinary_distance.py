"""
Calculates the distance beteen entities with trinary values using
Euclidean distances.
"""

import common_python.constants as cn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRINARY_VALUES = [-1, 0, 1]

class TrinaryDistance():

  def __init__(self, df_trinary):
    """
    Parameters
    ----------
    df_trinary: DataFrame
        columns: vector names
        index: instances
        values: -1, 0, 1
    """
    self.df_trinary = df_trinary
    # Validate the dataframe
    df_valid = self.df_trinary.applymap(lambda v: 1 if v in [-1, 0, 1]
        else np.nan)
    if np.isnan(df_valid.sum().sum()):
      raise ValueError("Argument is not a trinary dataframe")
    self.columns = df_trinary.columns
    self.num_col = len(self.columns)
    self.indices = self.df_trinary.index
    # Distance matrix
    #  columns: vector names
    #  index: vector names
    #  values: float
    self.df_distance = None

  def calcDistance(self):
    """
    Calculates the distance between all pairs of column vectors.
    """
    # Initialize the distance matrix
    arr = np.repeat(0, self.num_col)
    result_mat = np.repeat(arr, self.num_col)
    result_mat = np.reshape(result_mat, (self.num_col, self.num_col))
    trinary_mat = self.df_trinary.values
    for left_val in TRINARY_VALUES:
      left_func = lambda v: 1 if v==left_val else 0
      left_mat = np.transpose(np.vectorize(left_func)(trinary_mat))
      for right_val in TRINARY_VALUES:
        if left_val == right_val:
          continue
        right_func = lambda v: 1 if v==right_val else 0
        right_mat = np.vectorize(right_func)(trinary_mat)
        # Count the number of occurrences of this combination of values
        # by doing a matrix multiply
        new_mat = np.matmul(left_mat, right_mat)
        # Multiply by the squared distance between the values
        squared_distance = (left_val - right_val)**2
        new_mat = new_mat*squared_distance
        # Accumulate the result
        result_mat = result_mat + new_mat
    # Convert to dataframe
    self.df_distance = pd.DataFrame(result_mat, columns=self.columns,
        index=self.columns)

