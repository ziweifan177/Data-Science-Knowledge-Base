import threading as t
import multiprocessing as mp
import time

def job(q):
    res=0
    for i in range(1000):
        res+=i+i**2
    q.put(res)
    print(q)

def multicore(): #Process
    q=mp.Queue()
    p1=mp.Process(target=job, args=(q,))
    p2=mp.Process(target=job, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1=q.get()
    res2=q.get()
    print('multicore-res1: \n', res1)
    print('multicore-res2: \n', res2)

def multithread():
    q=mp.Queue()
    t1=t.Thread(target=job, args=(q,))
    t2=t.Thread(target=job, args=(q,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1=q.get()
    res2=q.get()

    print('multithread-res1', res1)
    print('multithread-res2', res2)

def normal():
    res=0
    for _ in range(2):
        for i in range(1000):
            res+=i+i**2
    print('normal-res:', res)

if __name__ =='__main__':
    normalTimeStart=time.time()
    normal()
    normalTimeEnd=time.time()
    print('\nNormal time:', normalTimeEnd-normalTimeStart) # 0.0015039443969726562

    multithread()
    mThreadTime = time.time()
    print('\nMultiThreadTime: ',mThreadTime-normalTimeEnd) # 0.03960561752319336

    multicore()
    mProcessTime = time.time()
    print('\nMultiProcessTime: ', mProcessTime-mThreadTime) #0.15290594100952148

