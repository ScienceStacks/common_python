"""Test helpers"""
import os
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

def getData():
  df_X = pd.read_csv(os.path.join(TEST_DIR, "feature_values.csv"))
  df_y = pd.read_csv(os.path.join(TEST_DIR, "class_values.csv"))
  df_X = df_X.set_index("index")
  df_y = df_y.set_index("index")
  ser_y = df_y["value"]
  return df_X, ser_y

def getDescription():
  df_desc = pd.read_csv(os.path.join(TEST_DIR, "feature_description.csv"))
  df_desc = df_desc.set_index("GENE_ID")
  ser_desc = df_desc["GO_Term"]
  return ser_desc
