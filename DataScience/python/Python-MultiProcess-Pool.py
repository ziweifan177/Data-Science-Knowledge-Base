import multiprocessing as mp

def job(x):
    return x*x

#Pool: Could return, no more Queue here.
def multicore(pool):
    res=pool.map(job, range(10))
    return(res)

#OR: animate the same effect with single process in the pool:
#pool.apply_async(): Must in iterations list.
def poolAsync(pool):
    #pool.apply_async(job,x): ONE core runs once with parameter x.
    multi_res=[pool.apply_async(job, (i,)) for i in range(10)] #Iterator.
    return[res.get() for res in multi_res]

if __name__=='__main__':
    pool = mp.Pool(processes=3)  # Processes: How many cores will be used? Default: all the cores will be used.
    print('multicore(pool): ', multicore(pool))
    print('poolAsync(pool): ', poolAsync(pool))
