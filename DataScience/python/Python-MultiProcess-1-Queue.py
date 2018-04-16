import multiprocessing as mp

#No return in process, just QUEUE(same with thread)

def job(queue):
    res=0
    for i in range(100):
        res+=i+i*2
    queue.put(res) #Return by queue.

if __name__=='__main__':
    q=mp.Queue()
    p1=mp.Process(target=job, args=(q,))
    p2=mp.Process(target=job, args=(q,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    result1=q.get()
    result2=q.get()

    print('result1: ',result1) #result1:  14850
    print('result2:', result2) #result2:  14850