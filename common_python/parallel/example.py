import threading
import time

def worker(num):
    """thread worker function"""
    print( 'Start %d' % num)
    time.sleep(2)
    print( 'Stop %d' % num)
    return

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()
