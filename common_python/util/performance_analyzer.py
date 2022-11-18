"""
Codes used to analyze performance.

Usage:
perf = PerformanceAnalyzer()

perf.start("outer")
for ...
    perf.start("one")
    # Codes for one
    ...
    perf.end("one")
perf.end("outer")
"""

import numpy as np
import pandas as pd
import time

SECS = 1e9  # Conversion from nano seconds


class PerformanceAnalyzer():

    def __init__(self):
        self.start_dct = {}  # Start time for a named segment
        self.elapsed_dct = {}  # Total time for each named segment

    def start(self, name):
        """
        Designates the start of a timer.

        Parameters
        ----------
        name: str
        """
        self.start_dct[name] = time.process_time_ns()

    def end(self, name):
        """
        Designates the end of a timer.

        Parameters
        ----------
        name: str
        """
        if not name in self.start_dct.keys():
            raise ValueError("%s has no start!" % name)
        if not name in self.elapsed_dct.keys():
            self.elapsed_dct[name] = 0
        self.elapsed_dct[name] += time.process_time_ns() \
              - self.start_dct[name]
        
    def report(self):
        """
        Assembles a report of elapsed times in seconds.

        Returns
        -------
        Series:
            index: name
            value: total time in seconds
        """
        ser = pd.Series(self.elapsed_dct)
        return ser/SECS
