'''
Created on Jun. 7, 2021

@author: zollen
'''

import threading

class MyThread(threading.Thread):
    
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        
    def run(self):
        
        global wlock
        
        wlock.acquire()
        
        for _ in range(1, 1000):
            print(self.threadID)
            
        wlock.release()
            

wlock = threading.Lock()
thread1 = MyThread(1, "Thread-1")
thread2 = MyThread(2, "Thread-2")      

threads = []
threads.append(thread1) 
threads.append(thread2)

for thread in threads:
    thread.start()
    
for thread in threads:
    thread.join()
    
print("DONE")