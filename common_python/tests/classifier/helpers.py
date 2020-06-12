"""Helpers for classifier testing."""

from common_python.util.persister import Persister
from common_python.classifier import feature_analyzer
from common_python.classifier  \
    import feature_set_collection

import os
import numpy as np
import pandas as pd



DIR_PATH = os.path.abspath(os.path.dirname(__file__))
# The PCL was constructed from TrinaryData
TEST_DATA_PATH = os.path.join(DIR_PATH,
    "test_classifier_data.pcl")
PERSISTER = Persister(TEST_DATA_PATH)
CLASS = 1
TEST_ANALYZER_PATH = os.path.join(DIR_PATH,
    "test_feature_analyzer_%d" % CLASS)
TEST_SER_COMB_PATH = os.path.join(DIR_PATH,
    "test_ser_comb_%d.csv" % CLASS)


if not PERSISTER.isExist():
  from common.trinary_data import TrinaryData
  DATA = TrinaryData()  # Will get an error if pcl not present
  DATA_LONG = TrinaryData(is_averaged=False,
      is_dropT1=False)
  # Will get an error if pcl not present
  PERSISTER.set([DATA, DATA_LONG])
else:
  try:
    [DATA, DATA_LONG] = PERSISTER.get()
  except:
    DATA = None
    DATA_LONG = None

def getData():
  """
  Provides classification data
  """
  df_X = DATA.df_X
  df_X.columns = DATA.features
  ser_y = DATA.ser_y
  return df_X, ser_y

def getDataLong():
  """
  Provides classification data
  """
  df_X = DATA_LONG.df_X
  df_X.columns = DATA_LONG.features
  ser_y = DATA_LONG.ser_y
  return df_X, ser_y

def getFeatureAnalyzer():
  analyzer = feature_analyzer.FeatureAnalyzer.deserialize(
      TEST_ANALYZER_PATH)
  return analyzer

def getCombSer():
  df = pd.read_csv(TEST_SER_COMB_PATH)
  index_col = df.columns.tolist()[0]
  value_col = df.columns.tolist()[1]
  ser = df[value_col].copy()
  ser.index = df[index_col]
  return ser
