import multiprocessing as mp

#Multiprocess with multi cores must be in main:
def job1(input):
    print('job1:', str(input))

if __name__=='__main__': #Multiprocess with multi cores must be in main:
    p1=mp.Process(target=job1,args=['Test In main'])
    p1.start()
    p1.join()

