import concurrent.futures
import logging
import time
import numpy as np

class MyThreads(object):

  def thread_function(self, args):
      name1, name2 = args
      print(name1, name2)
      name = name1 + "-" + name2
      logging.info("Thread %s: starting", name)
      time.sleep(2)
      logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
  format = "%(asctime)s: %(message)s"
  logging.basicConfig(format=format, 
      level=logging.INFO, datefmt="%H:%M:%S")

  my_threads = MyThreads()
  args = [('a', 'a'), ('b', 'b'), ('c', 'c')]
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=3) as executor:
    executor.map(my_threads.thread_function, args)
