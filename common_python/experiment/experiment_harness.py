"""
Harness for running computational experiments.

This is intended for computation studies where many combinations
of parameter values are explored. The harness requires:
  dictionary: parameter name, list of values
  function: a function that takes keyword arguments of parameters
            and returns a dataframe.
The result is a dataframe that is extended with columns
corresponding to the parameter values.
Intermediate values are stored in a file, allowing the
calculation to resume calculations. The file contains all values
when the calculation has completed.
"""

from common_python.util.multi_number import MultiNumber

import os
import pandas as pd
import numpy as np


OUT_PTH = "experiment_harness.csv"
UNNAMED = "Unnamed: 0"


class ExperimentHarness(object):

  def __init__(self, param_dct, func, out_pth=OUT_PTH,
      update_rpt=5):
    """
    :param dict param_dct: dictionary of parameter values
    :param Function func: function that takes keyword arguments
        of parameter values
    :param str out_pth: CSV file where intermediate
        results are stored
    :param int update_rpt: number of times the function
        is run before results are written.
    """
    self._out_pth = out_pth
    self._update_rpt = update_rpt
    self._func = func
    self.parameter_names = list(param_dct.keys())
    self.parameter_values = list(param_dct.values())
    # self.df_result - completed calculations
    # self._completeds - tuples of parameter values for whic
    #                    calculations have been completed
    self.df_result, self._completeds = self._makeRestoreDF()
    self._update_cnt = 0  # Position in reporting cycling

  def _makeRestoreDF(self):
    """
    Construct a dataframe of already completed calculations.
    :return pd.DataFrame, list:
    """
    if os.path.isfile(self._out_pth):
      df = pd.read_csv(self._out_pth)
      if UNNAMED in df.columns:
        del df[UNNAMED]
      if len(df) > 0:
        df_sub = df[self.parameter_names]
        df_sub = df_sub.drop_duplicates()
        completeds = [tuple(r) for _, r in df_sub.iterrows()]
    else:
      df = pd.DataFrame()
      completeds = []
    return df, completeds

  def _writeDF(self, is_force=False):
    """
    Writes self.df_result to the output file if required.
    :param pd.DataFrame df: dataframe to write
    :param bool is_force: force the write
    """
    self._update_cnt += 1
    if (self._update_cnt >= self._update_rpt) or is_force:
      self._update_cnt = 0
      self.df_result.to_csv(self._out_pth, index=False)

  def run(self):
    """
    Runs the function for the rest of the parameter combinations.
    :return pd.DataFrame:
    """
    # Initializations for evaluation
    bases = [len(v) for v in self.parameter_values]
    multi_number = MultiNumber(bases)  # List of positions in lists
    # Iterate
    for cur_pos in multi_number:
      values = [p[c] for p, c in 
          zip(self.parameter_values, cur_pos)]
      if not tuple(values) in self._completeds:
        kwargs = {k: v for k, v in zip(self.parameter_names, values)}
        # Calclate the dataframe for these values
        df = self._func(**kwargs)
        # Extend the dataframe with values of parameters used
        for k, v in kwargs.items():
          df[k] = np.repeat(v, len(df))
        self.df_result = pd.concat([self.df_result, df])
        # Write dataframe
        self._writeDF()
    # Completed
    self.df_result.index = list(range(len(self.df_result)))
    self._writeDF(is_force=True)
    return self.df_result
