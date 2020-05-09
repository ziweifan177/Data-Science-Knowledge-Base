import h2o
from h2o.automl import H2OAutoML
import random, os, sys
from datetime import datetime
import pandas as pd
import logging
import csv
import optparse
import time
import json
from distutils.util import strtobool
import psutil



parser=optparse.OptionParser()
parser.add_option('-a', '--analysis', help='analysis')
parser.add_option('-b', '--balance', help='balance classes')
parser.add_option('-c', '--classification', help='classification yes/no')
parser.add_option('-d', '--data', help='data file path')
parser.add_option('-e', '--min_mem_size', help='minimum memory size')
parser.add_option('-f', '--no_rows', help='number of rows')
parser.add_option('-g', '--nthreads', help='number of threads')
parser.add_option('-j', '--num_bad_levels', help='number of bad levels')
parser.add_option('-i', '--all_variables', help='all variables')
parser.add_option('-m', '--model', help='model file')
parser.add_option('-n', '--name', help='project name')
parser.add_option('-o', '--orig', help='original production data')
parser.add_option('-p', '--path', help='server path')
parser.add_option('-q', '--test', help='production data')
parser.add_option('-r', '--run_time', help='run time')
parser.add_option('-s', '--scale', help='scale yes/no')
parser.add_option('-t', '--target', help='dependent variable')
parser.add_option('-u', '--balance_threshold', help='balance threshold upsample')
parser.add_option('-x', '--rid', help='run id')
parser.add_option('-y', '--prod_rows', help='number of production rows')
(opts, args) = parser.parse_args()




# Functions

def scale_df(df,thresh=1000000):
    # df.shape[0]/x=thresh i.e int(thresh/df.shape[0])
    train_split=0.8
    test_split=0.1
    if thresh < df.shape[0]:
        train_split=round(thresh/df.shape[0],1)
        test_split=round((1-train_split)/2,1)   
    train, test, valid = df.split_frame([train_split, test_split])
    return  train, test, valid 

def alphabet(n):
    alpha='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'    
    str=''
    r=len(alpha)-1   
    while len(str)<n:
        i=random.randint(0,r)
        str+=alpha[i]   
    return str
  
def set_meta_data(no_rows,analysis,run_id,server,data,test,model_path,target,run_time,classification,scale,model,balance,balance_threshold,name,path,nthreads,min_mem_size):
    m_data={}
    m_data['no_rows'] = no_rows
    m_data['start_time'] = time.time()
    m_data['target']=target
    m_data['server_path']=server
    m_data['data_path']=data 
    m_data['test_path']=test
    m_data['max_models']=model
    m_data['run_time']=run_time
    m_data['run_id'] =run_id
    m_data['scale']=scale
    m_data['classification']=classification
    m_data['scale']=False
    m_data['model_path']=model_path
    m_data['balance']=balance
    m_data['balance_threshold']=balance_threshold
    m_data['project'] =name
    m_data['end_time'] = time.time()
    m_data['execution_time'] = 0.0
    m_data['run_path'] =path
    m_data['nthreads'] = nthreads
    m_data['min_mem_size'] = min_mem_size
    m_data['analysis'] = analysis
    return m_data  

def dict_to_json(dct,n):
    j = json.dumps(dct, indent=4)
    f = open(n, 'w')
    print(j, file=f)
    f.close()
  
  
def stackedensemble(mod):
    coef_norm=None
    try:
        metalearner = h2o.get_model(mod.metalearner()['name'])
        coef_norm=metalearner.coef_norm()
    except:
        pass        
    return coef_norm

def stackedensemble_df(df):
    bm_algo={ 'GBM': None,'GLM': None,'DRF': None,'XRT': None,'Dee': None}
    for index, row in df.iterrows():
        if len(row['model_id'])>3:
            key=row['model_id'][0:3]
            if key in bm_algo:
                if bm_algo[key] is None:
                    bm_algo[key]=row['model_id']
    bm=list(bm_algo.values()) 
    bm=list(filter(None.__ne__, bm))             
    return bm

def se_stats(modl):
    d={}
    try:    
        d['algo']=modl.algo
    except:
        pass      
    try:    
        d['model_id']=modl.model_id 
    except:
        pass      
    try:    
        d['null_degrees_of_freedom']=perf.null_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_degrees_of_freedom']=perf.residual_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_deviance']=perf.residual_deviance() 
    except:
        pass      
    try:    
        d['null_deviance']=perf.null_deviance() 
    except:
        pass      
    try:    
        d['aic']=perf.aic() 
    except:
        pass      
    try:
        d['logloss']=perf.logloss() 
    except:
        pass    
    try:
        d['auc']=perf.auc()
    except:
        pass  
    try:
        d['rmse']=modl.rmse()
    except:
        pass    
    try:
        d['roc']=modl.roc()
    except:
        pass  
    try:
        d['mse']=modl.mse()
    except:
        pass      
    return d

def get_model_by_algo(algo,models_dict):
    mod=None
    mod_id=None    
    for m in list(models_dict.keys()):
        if m[0:3]==algo:
            mod_id=m
            mod=h2o.get_model(m)      
    return mod,mod_id     
    
    
def gbm_stats(modl):
    d={} 
    try:    
        d['algo']=modl.algo
    except:
        pass      
    try:    
        d['model_id']=modl.model_id 
    except:
        pass  
    try:    
        d['varimp']=modl.varimp()
    except:
        pass           
    try:    
        d['null_degrees_of_freedom']=perf.null_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_degrees_of_freedom']=perf.residual_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_deviance']=perf.residual_deviance() 
    except:
        pass      
    try:    
        d['null_deviance']=perf.null_deviance() 
    except:
        pass      
    try:    
        d['aic']=perf.aic() 
    except:
        pass      
    try:
        d['logloss']=perf.logloss() 
    except:
        pass    
    try:
        d['auc']=perf.auc()
    except:
        pass  
    try:
        d['rmse']=modl.rmse()
    except:
        pass    
    try:
        d['roc']=modl.roc()
    except:
        pass  
    try:
        d['mse']=modl.mse()
    except:
        pass      
    return d
    
    
def dl_stats(modl):
    d={}

    try:    
        d['algo']=modl.algo
    except:
        pass      
    try:    
        d['model_id']=modl.model_id 
    except:
        pass  
    try:    
        d['varimp']=modl.varimp()
    except:
        pass           
    try:    
        d['null_degrees_of_freedom']=perf.null_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_degrees_of_freedom']=perf.residual_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_deviance']=perf.residual_deviance() 
    except:
        pass      
    try:    
        d['null_deviance']=perf.null_deviance() 
    except:
        pass      
    try:    
        d['aic']=perf.aic() 
    except:
        pass      
    try:
        d['logloss']=perf.logloss() 
    except:
        pass    
    try:
        d['auc']=perf.auc()
    except:
        pass  
    try:
        d['rmse']=modl.rmse()
    except:
        pass    
    try:
        d['roc']=modl.roc()
    except:
        pass  
    try:
        d['mse']=modl.mse()
    except:
        pass        
    return d
    
    
def drf_stats(modl):
    d={}

    try:    
        d['algo']=modl.algo
    except:
        pass      
    try:    
        d['model_id']=modl.model_id 
    except:
        pass  
    try:    
        d['varimp']=modl.varimp()
    except:
        pass           
    try:    
        d['null_degrees_of_freedom']=perf.null_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_degrees_of_freedom']=perf.residual_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_deviance']=perf.residual_deviance() 
    except:
        pass      
    try:    
        d['null_deviance']=perf.null_deviance() 
    except:
        pass      
    try:    
        d['aic']=perf.aic() 
    except:
        pass      
    try:
        d['logloss']=perf.logloss() 
    except:
        pass    
    try:
        d['auc']=perf.auc()
    except:
        pass  
    try:
        d['rmse']=modl.rmse()
    except:
        pass    
    try:
        d['roc']=modl.roc()
    except:
        pass  
    try:
        d['mse']=modl.mse()
    except:
        pass           
    return d
     
def xrt_stats(modl):
    d={}

    try:    
        d['algo']=modl.algo
    except:
        pass      
    try:    
        d['model_id']=modl.model_id 
    except:
        pass  
    try:    
        d['varimp']=modl.varimp()
    except:
        pass           
    try:    
        d['null_degrees_of_freedom']=perf.null_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_degrees_of_freedom']=perf.residual_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_deviance']=perf.residual_deviance() 
    except:
        pass      
    try:    
        d['null_deviance']=perf.null_deviance() 
    except:
        pass      
    try:    
        d['aic']=perf.aic() 
    except:
        pass      
    try:
        d['logloss']=perf.logloss() 
    except:
        pass    
    try:
        d['auc']=perf.auc()
    except:
        pass  
    try:
        d['rmse']=modl.rmse()
    except:
        pass    
    try:
        d['roc']=modl.roc()
    except:
        pass  
    try:
        d['mse']=modl.mse()
    except:
        pass           
    return d
    
def xgb_stats(modl):
    d={}

    try:    
        d['algo']=modl.algo
    except:
        pass      
    try:    
        d['model_id']=modl.model_id 
    except:
        pass  
    try:    
        d['varimp']=modl.varimp()
    except:
        pass           
    try:    
        d['null_degrees_of_freedom']=perf.null_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_degrees_of_freedom']=perf.residual_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_deviance']=perf.residual_deviance() 
    except:
        pass      
    try:    
        d['null_deviance']=perf.null_deviance() 
    except:
        pass      
    try:    
        d['aic']=perf.aic() 
    except:
        pass      
    try:
        d['logloss']=perf.logloss() 
    except:
        pass    
    try:
        d['auc']=perf.auc()
    except:
        pass  
    try:
        d['rmse']=modl.rmse()
    except:
        pass    
    try:
        d['roc']=modl.roc()
    except:
        pass  
    try:
        d['mse']=modl.mse()
    except:
        pass           
    return d
         
    
def glm_stats(modl):
    d={}

    try:    
        d['algo']=modl.algo
    except:
        pass      
    try:    
        d['model_id']=modl.model_id 
    except:
        pass  
    try:    
        d['varimp']=modl.varimp()
    except:
        pass           
    try:    
        d['null_degrees_of_freedom']=perf.null_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_degrees_of_freedom']=perf.residual_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_deviance']=perf.residual_deviance() 
    except:
        pass      
    try:    
        d['null_deviance']=perf.null_deviance() 
    except:
        pass      
    try:    
        d['aic']=perf.aic() 
    except:
        pass      
    try:
        d['logloss']=perf.logloss() 
    except:
        pass    
    try:
        d['auc']=perf.auc()
    except:
        pass  
    try:
        d['rmse']=modl.rmse()
    except:
        pass    
    try:
        d['roc']=modl.roc()
    except:
        pass  
    try:
        d['mse']=modl.mse()
    except:
        pass  
    try:
        d['coef']=modl.coef()
    except:
        pass  
    try:
        d['coef_norm']=modl.coef_norm() 
    except:
        pass            
    return d
    
def model_performance_stats(perf):
    d={}
    try:    
        d['mse']=perf.mse()
    except:
        pass      
    try:    
        d['rmse']=perf.rmse() 
    except:
        pass      
    try:    
        d['null_degrees_of_freedom']=perf.null_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_degrees_of_freedom']=perf.residual_degrees_of_freedom()
    except:
        pass      
    try:    
        d['residual_deviance']=perf.residual_deviance() 
    except:
        pass      
    try:    
        d['null_deviance']=perf.null_deviance() 
    except:
        pass      
    try:    
        d['aic']=perf.aic() 
    except:
        pass      
    try:
        d['logloss']=perf.logloss() 
    except:
        pass    
    try:
        d['auc']=perf.auc()
    except:
        pass  
    try:
        d['gini']=perf.gini()
    except:
        pass    
    return d
     
def impute_missing_values(df, x, scal=False):
    # determine column types
    ints, reals, enums = [], [], []
    for key, val in df.types.items():
        if key in x:
            if val == 'enum':
                enums.append(key)
            elif val == 'int':
                ints.append(key)            
            else: 
                reals.append(key)    
    _ = df[reals].impute(method='mean')
    _ = df[ints].impute(method='median')
    if scal:
        df[reals] = df[reals].scale()
        df[ints] = df[ints].scale()    
    return

def get_independent_variables(df, targ):
    C = [name for name in df.columns if name != targ]
    # determine column types
    ints, reals, enums = [], [], []
    for key, val in df.types.items():
        if key in C:
            if val == 'enum':
                enums.append(key)
            elif val == 'int':
                ints.append(key)            
            else: 
                reals.append(key)    
    x=ints+enums+reals
    return x
    
def get_all_variables_csv(i):
    ivd={}
    try:
        iv = pd.read_csv(i,header=None)
    except:
        logging.critical('pd.read_csv get_all_variables') 
        h2o.download_all_logs(dirname=logs_path, filename=logfile)
        h2o.cluster().shutdown()     
        sys.exit(9)             
    col=iv.values.tolist()[0]
    dt=iv.values.tolist()[1]
    i=0
    for c in col:
        ivd[c.strip()]=dt[i].strip()
        i+=1        
    return ivd
    
    

def check_all_variables(df,dct,y=None):     
    targ=list(dct.keys())     
    for key, val in df.types.items():
        if key in targ:
            if dct[key] not in ['real','int','enum']:                      
                targ.remove(key)  
    for key, val in df.types.items():
        if key in targ:            
            if dct[key] != val:
                print('convert ',key,' ',dct[key],' ',val)
                if dct[key]=='enum':
                    try:
                        df[key] = df[key].asfactor() 
                    except:
                        targ.remove(key)                 
                if dct[key]=='int': 
                    try:                
                        df[key] = df[key].asnumeric() 
                    except:
                        targ.remove(key)                  
                if dct[key]=='real':
                    try:                
                        df[key] = df[key].asnumeric()  
                    except:
                        targ.remove(key)                  
    if y is None:
        y=df.columns[-1] 
    if y in targ:
        targ.remove(y)
    else:
        y=targ.pop()            
    return targ    
    

def predictions(mod,test,test_X,run_id,allV,slice_no=0):
   # test = h2o.import_file(data)
    
    if slice_no>0:
        test = test[0:slice_no,:]
    
    if allV is not None:
        ivd=get_all_variables_csv(allV)    
        X=check_all_variables(test, ivd, y)
    
    mod_perf=mod_best.model_performance(test)
              
    stats_test={}
    stats_test=model_performance_stats(mod_perf)

    n=run_id+'_test_stats.json'       
    dict_to_json(stats_test,n) 
    try:    
        cf=mod_perf.confusion_matrix(metrics=["f1","f2","f0point5","accuracy","precision","recall","specificity","absolute_mcc","min_per_class_accuracy","mean_per_class_accuracy"])
        cf_df=cf[0].table.as_data_frame()
        cf_df.to_csv(run_id+'_test_confusion_matrix.csv')
    except:
        pass

    predictions = mod_best.predict(test_X)
    predictions_df=test_X.cbind(predictions)	# My changes    
 #   predictions_df=test.cbind(predictions)	# My changes
    n=run_id+'_predictions.csv'     
    h2o.export_file(predictions_df, n)       # My changes    

    return
    



def check_X(x,df):
    for name in x:
        if name not in df.columns:
            x.remove(name)  
    return x    
    
def check_y(y,df):
    ok=False
    C = [name for name in df.columns if name == y]
    for key, val in df.types.items():
        if key in C:
            if val in ['real','int','enum']:        
                ok=True     # Add    
    return ok     
    
def get_stacked_ensemble(lst):
    se=None
    for model in model_set:
        if 'BestOfFamily' in model:
            se=model
    if se is None:     
        for model in model_set:
            if 'AllModels'in model:
                se=model           
    return se       
    
def get_variables_types(df):
    d={}
    for key, val in df.types.items():
        d[key]=val           
    return d      
    
    
    
def mod_best_stats(mod,run_id):           
    stats_mod={}
    try:
        stats_mod['accuracy']=mod.accuracy()
    except:
        pass 
    try:
        stats_mod['F1']=mod.F1()[0]  
    except:
        pass 
    try:
        stats_mod['F2']=mod.F2()[0]
    except:
        pass 
    try:
        stats_mod['aic']=mod.aic()
    except:
        pass 
    try:
        stats_mod['algo']=mod.algo
    except:
        pass 
    try:
        stats_mod['auc']=mod.auc()
    except:
        pass 
    try:
        stats_mod['error']=mod.error()
    except:
        pass 
    try:
        stats_mod['gini']=mod.gini()
    except:
        pass 
    try:
        stats_mod['logloss']=mod.logloss()
    except:
        pass 
    try:
        stats_mod['mean_per_class_error']=mod.mean_per_class_error()
    except:
        pass 
    try:
        stats_mod['mse']=mod.mse()
    except:
        pass 
    try:
        stats_mod['precision']=mod.precision()
    except:
        pass 
    try:
        stats_mod['rmse']=mod.rmse()
    except:
        pass 
    
    n=run_id+'_mod_best_stats.json'
    dict_to_json(stats_mod,n) 
    
    try:             
        cf=mod.confusion_matrix(metrics=["f1","f2","f0point5","accuracy","precision","recall","specificity","absolute_mcc","min_per_class_accuracy","mean_per_class_accuracy"])
        cf_df=cf[0].table.as_data_frame()
        cf_df.to_csv(run_id+'_mod_best_confusion_matrix.csv')   
    except:   
        pass

    return    
        
    

def variables_diff(df1,df2,t):
    v1=get_variables_types(df1)
    v2=get_variables_types(df2) 
    iv={}  
    dv={} 
    if t in v1:
        iv[t]=v1[t]    
    for k, v in v1.items():
        if k in v2:    
            if v2[k]==v:
                iv[k]=v 
            else:
                l=[v,v2[k]]           
                dv[k]=l        
    return iv,  dv, v1, v2

def get_index_columns(l,d):
    i=0
    idx=[]
    for n in l:
        if n in d.keys():    
            idx.append(i)
        i=i+1   
    return idx 
    
def remove_variables(dft,dct):  
    targ=list(dct.keys())
   # dft.drop(0)
    for key, val in dft.types.items():
        if key in targ:
            if dct[key] not in ['real','int','enum']:                      
                targ.remove(key)  
    for key, val in df.types.items():      
        if key not in targ:  
            drop=True    
            try:  
                if dct[key] in ['real','int']:                   
                    dft[key] = dft[key].asnumeric()          
                if dct[key] in ['enum']:                   
                    dft[key] = dft[key].asfactor()           
                drop=False             
            except:
                pass 
            if drop:
                try:                                
                    dft.drop(key)   
                except:
                    pass                
    return dft 
 
def remove_nums(s):
    l = [x.strip() for x in s.split(',')] 
    nums=[]
    for i in l:
        try:
            nums.append(int(i))   
        except:
            pass
    return nums
  
def good_idx(l,hf):
    idx=list(range(hf.shape[0]))
    idx=[x for x in idx if x not in l]
    return idx
      
        
#  End Functions

min_mem_size=6
pct_memory=0.5
virtual_memory=psutil.virtual_memory()
max_mem_size=int(round(int(pct_memory*virtual_memory.available)/1073741824,0))
if min_mem_size > max_mem_size:
    min_mem_size=max_mem_size  
# if True:

try:


# auto reducing of the data 
    no_rows=1000000
    if opts.no_rows is not None:
        try:
            no_rows=int(opts.no_rows)  # set no_rows
        except:
            sys.exit(5)          

# number of production rows 
    prod_rows=0
    if opts.prod_rows is not None:
        try:
            prod_rows=int(opts.prod_rows)  # set no_rows
        except:
            sys.exit(5)      
     
    data_path=None
    if opts.data is not None:
        try:
            data_path=str(opts.data)  # data_path = '/Users/bear/Downloads/H2O/VD/data/loan.csv'
        except:
            sys.exit(5) 
     
    orig_path=None
    if opts.orig is not None:
        try:
            orig_path=str(opts.orig)  # parser.add_option('-o', '--orig', help='original production data')
        except:
            sys.exit(5)    
     
    all_variables=None
    if opts.all_variables is not None:
        try:
            all_variables=str(opts.all_variables)  # data_path = '/Users/bear/Downloads/H2O/VD/data/loan.fields.csv'
        except:
            sys.exit(5) 
   
    test_path=None
    if opts.test is not None:
        try:
            test_path=str(opts.test)  # test = '/Users/bear/Downloads/H2O/VD/data/loan.test.csv'
        except:
            sys.exit(5)   
     
    rid=None
         
    if opts.rid is not None:
        try:
            rid=str(opts.rid)  # test = '/Users/bear/Downloads/H2O/VD/data/loan.test.csv'
        except:
            sys.exit(5) 
        
    analysis=5
    if opts.analysis is not None:
        try:
            analysis=int(opts.analysis)  # test = '/Users/bear/Downloads/H2O/VD/data/loan.test.csv'
        except:
            sys.exit(5) 
   
    target=None
    if opts.target is not None:
        try:
            target=str(opts.target)  # target = 'dependent_variable'
        except:
            sys.exit(5)
        
            
    nthreads=1 
    if opts.nthreads is not None:
        try:
            nthreads=int(opts.nthreads)  
        except:
            sys.exit(5) 
        
    # min_mem_size=int(round(int(pct_memory*virtual_memory.available)/1073741824,0))
    
    if opts.min_mem_size is not None:
        try:
            min_mem_size=int(opts.min_mem_size)  
        except:
            sys.exit(5)     
        
    # run_time=360       
    run_time=360
    if opts.run_time is not None:
        try:
            run_time=int(opts.run_time)  # run_time=360
        except:
            sys.exit(5) 
        
    # num_bad_levels=0      
    num_bad_levels=0 
    
    if opts.num_bad_levels is not None:
        try:
            num_bad_levels=int(opts.num_bad_levels)  # num_bad_levels=0   
        except:
            sys.exit(5)       
           
          
    '''
    True values are y, yes, t, true, on and 1; false values are n, no, f, false, off and 0. Raises ValueError if val is anything else.
    This returns 1/0 not True/False, so you need to wrap the result in bool() to get actual boolean: bool(distutils.util.strtobool(some_string))
    '''
    
    classification=False
    if opts.classification is not None:
        try:
            classification=bool(strtobool(opts.classification))  # classification=True
        except:
            sys.exit(5) 
        
        
    scale=False
    if opts.scale is not None:
        try:
            scale=bool(strtobool(opts.scale))  # target = 'dependent_variable'
        except:
            sys.exit(5) 
        
    max_models=9    
    model_path=None
    if opts.model is not None:
        try:
            model_path=str(opts.model)  # 
        except:
            sys.exit(5) 
        
        
    balance_y=False 
    if opts.balance is not None:
        try:
            balance=bool(strtobool(opts.balance))  # balance=True 
        except:
            sys.exit(5) 

    balance_threshold=0.2
    if opts.balance_threshold is not None:
        try:
            balance_threshold=str(opts.balance_threshold)  # balance_threshold=0.2
        except:
            sys.exit(5) 
        
    name=None 
    if opts.name is not None:
        try:
            project=str(opts.name)  # project='project name'
        except:
            sys.exit(5) 
        
    server_path=None
    if opts.path is not None:
        try:
            server_path=str(opts.path)  # server_path='/Users/bear/Downloads/H2O/VD'
        except:
            sys.exit(5) 
    
    #test_path=prod_path
    print(test_path)
    
    # Automodeler start
    
    if rid is None:      
        run_id=alphabet(9)
    else:      
        run_id=rid   
    if server_path==None:
        server_path=os.path.abspath(os.curdir)
    os.chdir(server_path) 
    run_dir = os.path.join(server_path,run_id)
    os.mkdir(run_dir)
    os.chdir(run_dir)    
    
    print (run_id) # run_id to std out
    start = datetime.now()
    
    
    # Logs
    
    logfile=run_id+'_autoh2o_log.zip'
    logs_path=os.path.join(run_dir,'logs')
    
    
    # logging
    log_file=run_id+'.log'
    log_file = os.path.join(run_dir,log_file)
    logging.basicConfig(filename=log_file,level=logging.INFO,format="%(asctime)s:%(levelname)s:%(message)s")
    logging.info(start) 
    
    
    print(min_mem_size,' ',max_mem_size)
    
    # 65535 Highest port no
    port_no=random.randint(5555,55555)
    
    #  h2o.init(strict_version_check=False,min_mem_size_GB=min_mem_size,port=port_no) # start h2o
    
    h2o.init(strict_version_check=False,min_mem_size_GB=min_mem_size,max_mem_size_GB=max_mem_size,port=port_no) # start h2o
    
    # meta data
    meta_data = set_meta_data(no_rows,analysis,run_id,server_path,data_path,test_path,model_path,target,run_time,classification,scale,max_models,balance_y,balance_threshold,name,run_dir,nthreads,min_mem_size)
    
    
    # predictions only
    if model_path is not None:
        if test_path is None:    
            logging.info('model_path no test_path')
            h2o.download_all_logs(dirname=logs_path, filename=logfile)         
            h2o.cluster().shutdown()
            sys.exit(1)     
        # Load model  mod_best     
        mod_best = h2o.load_model(model_path)
        _=predictions(mod_best,test_path,run_id)
      
        meta_data['end_time'] = time.time()
        meta_data['execution_time'] = meta_data['end_time'] - meta_data['start_time']
        
        n=run_id+'_meta_data.json'
        dict_to_json(meta_data,n)  
      
        os.chdir(server_path)
      
        h2o.cluster().shutdown() 
        sys.exit(0)    
    
    print(data_path)
    
    try:
        df = h2o.import_file(data_path)
    except:
        logging.critical('h2o.import_file(data_path)')    
        h2o.download_all_logs(dirname=logs_path, filename=logfile)    
        h2o.cluster().shutdown()
        sys.exit(3)
    
    
    
    # %tb
    df.head()
    
    df.describe()
    
    df.columns
    
    # Med_Paid, Med Incurred, Incurred_Collection, Claim_Total_Reserve, Claim_Total_Paid, Claim_Total_Incurred
    # df[['ProductId',
    #  'UserId',
    #  'Summary',
    #  'Score',
    #  'HelpfulnessDenominator',
    #  'Id',
    #  'ProfileName',
    #  'HelpfulnessNumerator',
    #  'Time',
    #  'Text',
    #  'TextC1',
    #  'TextC2',
    #  'TextC3',
    #  'TextC4',
    #  'TextC5',
    #  'TextC6',
    #  'TextC7',
    #  'TextC8',
    #  'TextC9',
    #  'TextC10']].describe()
    
    df[df.columns].describe()
    
    df.shape
    
    
    if target==None:
        target=df.columns[-1]   
    y = target
    print(y)
    
    print(test_path)
    
    # Load and analyze production versus training files
    
    prod=None  
    
    
    if test_path is not None:
        try:  
            prod = h2o.import_file(test_path,header=1)  
        except:
            logging.critical('h2o.import_file(test_path)')    
            h2o.download_all_logs(dirname=logs_path, filename=logfile)    
            h2o.cluster().shutdown()
            sys.exit(3)
      
    if test_path is not None:
        print(prod)
        prod.head()
    
    
    if prod is not None:
        print(prod.shape)
    
    df.columns
    
    df.columns.remove(target)
    
    # Med_Paid, Med Incurred, Incurred_Collection, Claim_Total_Reserve, Claim_Total_Paid, Claim_Total_Incurred
    if prod is not None:
        prod[['Name',
              'Sex',
              'Age',
              'SibSp',
              'Parch',
              'Ticket',
              'Fare',
              'Cabin'
            ]].describe()
    
    # auto reducing of the data     
    # Cut data to roughly no_rows if data have more rows than no_rows
    pct_rows=round(no_rows/df.shape[0],1)
    train, test, valid = [[],[],[]]
    if pct_rows < 1:
        half_pct=round(((1-pct_rows)/2),1)  
        df, test, valid = df.split_frame([pct_rows,half_pct])
    
    # assign target and inputs for classification or regression
    if target==None:
        target=df.columns[-1]   
    y = target
    print(y)
    
    # get variables types  
    allV, diffV, v1, v2 = {},{},{},{}
       
    if test_path is not None:
        allV, diffV, v1, v2 = variables_diff(df,prod,y)
    else:
        allV=get_variables_types(df)    
    
    allV
    
    diffV
    
    v1
    
    v2
    print(y)
    
    meta_data['variables']=allV
    meta_data['diff_variables']=diffV
    meta_data['train_variables']=v1
    meta_data['production_variables']=v2
    
    
    # Remove unused independent variables
    
    df=remove_variables(df,allV)
    idx=get_index_columns(df.columns,allV)  
    df = df[:,idx]
    
    
    
    if prod is not None:  
        prod=remove_variables(prod,allV)
        idx=get_index_columns(prod.columns,allV)  
        prod = prod[:,idx] 
    
    print(df.columns)
    
    #all_variables='/Users/bear/Downloads/code/DAT/titanic_variables-1322-1.csv'
    
    # independent variables
    
    X = []  
    
    X=get_independent_variables(df, target) 
    
    print(X)
    
    if all_variables is None:
        X=get_independent_variables(df, target)  
    else: 
        ivd=get_all_variables_csv(all_variables)    
        X=check_all_variables(df,ivd,y)
    
    X=check_X(X,df)
    
    print(X)
    
    # Add independent variables
    
    meta_data['X']=X  
    
    
    # impute missing values
    try:
        _=impute_missing_values(df,X, scale)
    except:
        pass
    
    # Force target to be factors
    # Only 'int' or 'string' are allowed for asfactor(), got Target (Total orders):real 
    
    if analysis == 3:
        classification=False
    elif analysis == 2:
        classification=True
    elif analysis == 1:
        classification=True  
    
    if classification:
        df[y] = df[y].asfactor()
        print('classification')    
    else:
        print('regression')     
    ok=check_y(y,df)  
    
    if not ok:  
        logging.critical('check_y') 
        h2o.download_all_logs(dirname=logs_path, filename=logfile)      
        h2o.cluster().shutdown()   
        sys.exit(9)  
    
    
    #%tb
    
    print(y)
    
    df[y].levels()
    
    print(X)
    
    print(classification)
    
    print(y)
    
    df.head()
    
    df.describe()
    
    # Set up AutoML
    
    aml = H2OAutoML(max_runtime_secs=run_time,project_name = name)
    
    
    # t appears that by removing the Med_Paid, Med Incurred, Incurred_Collection, Claim_Total_Reserve, Claim_Total_Paid, Claim_Total_Incurred columns the Knee dataset runs? 
    # 
    # What is it about these columns?
    
    
    X
    
    
    Xy=[y]
    
    
    Xy
    
    
    for e in X:
        Xy.append(e)
    
    Xy
    
    
    X
    
    
    train_df=df[:,Xy]
    
    
    train_df.shape
    
    
    df.shape
    
    
    # # train model
    # 
    # model_start_time = time.time()
    #   
    # try:
    #   aml.train(x=X,y=y,training_frame=df)  # Change training_frame=train
    # except Exception as e:
    #   logging.critical('aml.train') 
    #   h2o.download_all_logs(dirname=logs_path, filename=logfile)      
    #   h2o.cluster().shutdown()   
    #   sys.exit(4)
    #   
    # meta_data['model_execution_time'] = time.time() - model_start_time  
    
    model_start_time = time.time()
    aml.train(x=X,y=y,training_frame=train_df)  # Change training_frame=train
    '''
      
    try:
      aml.train(x=X,y=y,training_frame=train_df)  # Change training_frame=train
    except Exception as e:
      logging.critical('aml.train') 
      h2o.download_all_logs(dirname=logs_path, filename=logfile)      
      h2o.cluster().shutdown()   
      sys.exit(4)
      
    '''  
      
    meta_data['model_execution_time'] = time.time() - model_start_time 
    
    
    # %tb
    
    
    aml.leaderboard
    
    # get leaderboard
    aml_leaderboard_df=aml.leaderboard.as_data_frame()
    
    
    # STart best model as first model
    
    model_set=aml_leaderboard_df['model_id']
    
    
    meta_data['no_models']=0
    
    
    try:  
        mod_best=h2o.get_model(model_set[0])
    except Exception as e:
        logging.critical('ml_leaderboard_df_model_id_0') 
        h2o.download_all_logs(dirname=logs_path, filename=logfile)      
        h2o.cluster().shutdown()   
        sys.exit(4)
    
    # Get stacked ensemble  
    se=get_stacked_ensemble(model_set)
    
    
    # Stacked Ensemble stats 
    
    if se is not None:
        mod_best=h2o.get_model(se)
    #  print(mod_best.algo)    
        coef_norm={}
        try:     
            metalearner = h2o.get_model(mod_best.metalearner()['name'])
            coef_norm=metalearner.coef_norm() 
       #     print(coef_norm)      
        except:
            pass 
       
        try:         
            stats_se={}      
            stats_se=se_stats(mod_best)
            stats_se['coef_norm']=coef_norm     
            n=run_id+'_se_stats.json'
            dict_to_json(stats_se,n)
       #     print(stats_se)            
        except:
            pass 
    
    
    #  Get best_models and coef_norm()
    best_models={}
    best_models=stackedensemble(mod_best)
    bm=[]
    if best_models is not None: 
        if 'Intercept' in best_models.keys():
            del best_models['Intercept']
        bm=list(best_models.keys())
    else:
        best_models={}
        bm=stackedensemble_df(aml_leaderboard_df)   
        for b in bm:   
            best_models[b]=None
    
    if mod_best.model_id not in bm:
        bm.append(mod_best.model_id)
    
    
    # Mod best stats 
    
    try:
        mod_best_stats(mod_best,run_id)
    except:
        pass   
    
    
    mod_best_stats(mod_best,run_id)
    
    mod_best
    
    if classification:
        try:
            cf=mod_best.confusion_matrix(metrics=["f1","f2","f0point5","accuracy","precision","recall","specificity","absolute_mcc","min_per_class_accuracy","mean_per_class_accuracy"])
            print(cf)
            cf_df=cf[0].table.as_data_frame()
            cf_df.to_csv(run_id+'_mod_best_confusion_matrix.csv')
        except:
            pass
    
    # Best of Family leaderboard
    
    aml_leaderboard_df=aml_leaderboard_df.loc[aml_leaderboard_df['model_id'].isin(bm)]
    
    aml_leaderboard_df
    
    meta_data['no_models']=len(aml_leaderboard_df)
    
    
    # save leaderboard
    leaderboard_stats=run_id+'_leaderboard.csv'
    aml_leaderboard_df.to_csv(leaderboard_stats)
    
    # best model is at top of leader board
    top=aml_leaderboard_df.iloc[0]['model_id']  
    mod_best=h2o.get_model(top)
    
    # Add best model to meta file 
    meta_data['mod_best']=mod_best._id
    meta_data['mod_best_algo']=mod_best.algo  
    
    # save models
    try:
        models_path=os.path.join(run_dir,'models')
        for mod in bm:
            try:   
                m=h2o.get_model(mod) 
                h2o.save_model(m, path = models_path)
            except:    
                pass 
    except Exception as e:  
        logging.critical('save_model')  
        h2o.download_all_logs(dirname=logs_path, filename=logfile)      
        h2o.cluster().shutdown() 
        sys.exit(7)
    
    
    # GBM
     
    mod,mod_id=get_model_by_algo("GBM",best_models)
    if mod is not None:
        try:     
            sh_df=mod.scoring_history()
            sh_df.to_csv(run_id+'_gbm_scoring_history.csv') 
        except:
            pass   
        try:     
            stats_gbm={}
            stats_gbm=gbm_stats(mod)
            n=run_id+'_gbm_stats.json'
            dict_to_json(stats_gbm,n)
          #  print(stats_gbm)
        except:
            pass  
    
    # DeepLearning
    mod,mod_id=get_model_by_algo("Dee",best_models)
    if mod is not None:
        try:    
            sh_df=mod.scoring_history()
            sh_df.to_csv(run_id+'_dl_scoring_history.csv') 
        except:
            pass 
        try:
            stats_dl={}
            stats_dl=dl_stats(mod)
            n=run_id+'_dl_stats.json'
            dict_to_json(stats_dl,n)
          #  print(stats_dl)
        except:
            pass    
        try:
            cf=mod.confusion_matrix()    
            cf_df.to_csv(run_id+'_dl_confusion_matrix.csv')
        except:
            pass   
    
    
    # DRF
    mod,mod_id=get_model_by_algo("DRF",best_models)
    if mod is not None:
        try:     
            sh_df=mod.scoring_history()
            sh_df.to_csv(run_id+'_drf_scoring_history.csv') 
        except:
            pass  
        try: 
            stats_drf={}
            stats_drf=drf_stats(mod)
            n=run_id+'_drf_stats.json'
            dict_to_json(stats_drf,n)
            #print(stats_drf)
        except:
            pass     
    
    
    # XRT
    mod,mod_id=get_model_by_algo("XRT",best_models)
    if mod is not None:
        try:     
            sh_df=mod.scoring_history()
            sh_df.to_csv(run_id+'_xrt_scoring_history.csv')
        except:
            pass     
        try:        
            stats_xrt={}
            stats_xrt=xrt_stats(mod)
            n=run_id+'_xrt_stats.json'
            dict_to_json(stats_xrt,n)
         #   print(stats_xrt)
        except:
             pass    
    
    # GLM
    mod,mod_id=get_model_by_algo("GLM",best_models)
    if mod is not None:
        try:     
            stats_glm={}
            stats_glm=glm_stats(mod)
            n=run_id+'_glm_stats.json'
            dict_to_json(stats_glm,n)
          #  print(stats_glm)
        except:
            pass  
    
    
    # XGBOost
    
    mod,mod_id=get_model_by_algo("XGB",best_models)  # XGBoost_1_AutoML_20190117_232311
    if mod is not None:
        try:     
            sh_df=mod.scoring_history()
            sh_df.to_csv(run_id+'_xgb_scoring_history.csv')
        except:
            pass     
        try:        
            stats_xgb={}
            stats_xgb=xgb_stats(mod)
            n=run_id+'_xgb_stats.json'
            dict_to_json(stats_xgb,n)
         #   print(stats_xgb)
        except:
            pass 
             
    
    
    # ##### If test data then create predictions
    # 
    # if test_path is not None:
    #   try:
    #     _=predictions(mod_best,test_path,run_id,all_variables)
    #   except Exception as e:
    #     print(e)  
    #     logging.critical('h2o predictions')
    #     h2o.download_all_logs(dirname=logs_path, filename=logfile)        
    #     h2o.cluster().shutdown()
    #     sys.exit(9)
    
    prod.shape
    
    prod_X=prod[:,X]
    
    X
    
    prod_X.shape
    
    prod_X.columns
    
    # If production data then create predictions
    prediction_error = 0  
    if test_path is not None:
        try:
            _=predictions(mod_best,prod,prod_X,run_id,all_variables,prod_rows)
        except Exception as e:
            print(e) 
            prediction_error=1 
            logging.info('h2o predictions')
            h2o.download_all_logs(dirname=logs_path, filename=logfile) 
    
    print(prediction_error)
    
    pred_run=[prod_rows,prediction_error]
    meta_data['nprediction_run']=pred_run
    
    # Update and save meta data
    
    meta_data['end_time'] = time.time()
    meta_data['execution_time'] = meta_data['end_time'] - meta_data['start_time']
      
    n=run_id+'_meta_data.json'
    dict_to_json(meta_data,n)    
    
    # Save logs
    h2o.download_all_logs(dirname=logs_path, filename=logfile)
    
    os.chdir(server_path)
    
    h2o.cluster().shutdown()
    sys.exit(0)
    
except Exception as e:
    print(e)  
    logging.critical('h2o.unknown')
    h2o.download_all_logs(dirname=logs_path, filename=logfile)   
    h2o.cluster().shutdown()
    sys.exit(9)    