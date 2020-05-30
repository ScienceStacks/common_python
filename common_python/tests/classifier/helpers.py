"""Helpers for classifier testing."""

from common_python.util.persister import Persister
from common.trinary_data import TrinaryData
#from common.trinary_data import TrinaryData

import os



DIR_PATH = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_PATH = os.path.join(DIR_PATH,
    "test_classifier_data.pcl")
PERSISTER = Persister(TEST_DATA_PATH)


if not PERSISTER.isExist():
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
