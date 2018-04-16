import threading

#1. threading foundation:
def test():
    print('threading.active_count():',threading.active_count()) # Current threading number.
    print('threading.enumerate():',threading.enumerate())#What the specific thread?
    print('threading.current_thread():',threading.current_thread()) #Current threading.

    print('1. threading foundation:')
    test()
    #Output: 1
    #Output: [<_MainThread(MainThread, started 7488)>]
    #Output: <_MainThread(MainThread, started 7808)>

#2. threading: Add new thread:
def thread_job():
    print('This is added thread: num is %s' % threading.current_thread()) #OUTPUT:<Thread(Thread-1, started 1772)>

def AddNewThread():
    added_thread=threading.Thread(target=thread_job)#Add new thread.
    added_thread.start()#Start this new thread.

    print('threading.active_count():', threading.active_count()) #OUTPUT: 2
    print('threading.enumerate():', threading.enumerate())  #OUTPUT: [<_MainThread(MainThread, started 20092)>]
    print('threading.current_thread():', threading.current_thread())  # OUTPUT:<_MainThread(MainThread, started 20092)>

print('\n2. threading-Add new thread:')
AddNewThread()

#3.Thread_join():
import time

def addThread1():
    print('Thread 1 start! number is %s' % threading.current_thread())
    for i in range(10):
        time.sleep(0.1)
    print('Thread 1 done.')

def addThread2():
    print('Thread 2 start! Number is %s' % threading.current_thread())
    print('Thread 2 done.')

def threadsWork():
    thread_1=threading.Thread(target=addThread1, name='thread1')
    thread_1.start()
    thread_1.join() #Wait for Thread1 done! then go ahead.

    thread_2=threading.Thread(target=addThread2(),name='thread2')
    thread_2.start()
    print('main done.')

print('\n3.Thread_join():')
threadsWork()

#4. Threading-queue: No return in thread, just queue!
from queue import Queue

def thread_job(list,queue):
    for i in range(len(list)):
        list[i]=list[i]**2
    queue.put(list) #No return here! just queue.

def createMultiThread(data):
    q=Queue() #Queue: Store the result of every queue.
    threadList=[] #List: Store threads.

    for i in range(4):
        thread=threading.Thread(target=thread_job, args=(data[i],q))
        thread.start()
        threadList.append(thread)

    for thread in threadList:
        thread.join()

    return q

def getResult(queue):
    results=[]
    for _ in range(4):
        results.append(queue.get())
    print(results)

print('\n4. Threading-queue:')
list=[[1,2,3],[3,4,5],[5,2,6],[7,4,2]]
q=createMultiThread(list)
getResult(q)

#5. Thread-Lock:线程不仅仅可以是一个短function，也可是是一个长的，
# 而在执行长功能的时候，可以对长功能中的一小部分实现lock,这样只有那一小部分不会互相影响﻿。
#Compare with join()： lock 是锁住变量的更新, join 是不让主线程比某个 thread 先结束﻿
def job1():
    global A,lock
    lock.acquire()
    for i in range(10):
        A+=10
        print('Job1:',A)
    lock.release()

def job2():
    global A,lock
    lock.acquire()
    for i in range(10):
        A+=10
        print('Job2:',A)
    lock.release()

print("\n5. Thread-Lock:")
lock=threading.Lock()
A=0
thread1=threading.Thread(target=job1)
thread2=threading.Thread(target=job2)
thread1.start()
thread1.join()
thread2.start()
thread2.join()




