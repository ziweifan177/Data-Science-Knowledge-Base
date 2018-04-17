import multiprocessing as mp
import time

def job(lock,v,delta):
    lock.acquire() #Lock this job!
    for _ in range(10):
        time.sleep(0.1)
        v.value+=delta
        print(v.value)
    lock.release() #Release the job!

def multicore():
    lock=mp.Lock()
    v=mp.Value('i',0) #Shared memory:i=0
    p1=mp.Process(target=job, args=(lock,v,1))
    p2=mp.Process(target=job, args=(lock,v,3))
    p1.start()
    p2.start()

    p1.join()
    p2.join()

if __name__=='__main__':
    multicore()

