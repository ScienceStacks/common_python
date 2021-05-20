"""Time Code Segments"""

import time

class Timer():

    def __init__(self, name, isEnable=True):
        self.name = name
        self.isEnable = isEnable
        if self.isEnable:
            self.startTime = time.time()

    def start(self):
        """
        Start timing.
        """
        if self.isEnable:
            self.startTime = time.time()

    def print(self, name=None):
        """
        Print elapsed time since last start. Restarts the timer.

        Parameters
        ----------
        name: str
            name printed with the timer report
        """
        if not self.isEnable:
            return
        if name is None:
            name = self.name
        elapsed = time.time() - self.startTime
        print("***%s: %2.4f" % (name, elapsed))
        self.start()
