import threading
import time

lock = threading.Lock()
is_locking = False

def worker(num):
    """thread worker function"""
    print( 'Start %d' % num)
    if is_locking:
      lock.acquire()
    time.sleep(2)
    if is_locking:
      lock.release()
    print( 'Stop %d' % num)
    return

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()
