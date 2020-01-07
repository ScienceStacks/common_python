"""Utilities for access databases, especially sqlite."""

import common_python.constants as cn

import os
import pandas as pd
import numpy as np
import sqlite3


def csvToTable(csv_path, db_path, tablename=None):
  """
  Write a CSV file as a table name, overwriting if exists.
  :param str csv_path: path to the CSV file
  :param str db_path: path to the database
  :parm str tablename:
  """
  df = pd.read_csv(csv_path)
  conn = sqlite3.connect(db_path)
  if tablename is None:
    filename = os.path.split(csv_path)[1]
    tablename = os.path.splitext(filename)[0]
  df.to_sql(tablename, conn, if_exists='replace')

