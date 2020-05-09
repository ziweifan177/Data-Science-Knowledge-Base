# python3 /home/nik/data.preprocessing.py --target='bad_loan' --data='/home/nik/data/loan.csv' 
# python3 /home/nik/data.preprocessing.py --data='/home/nik/data/TESTTRAININGDATA2.csv' 
# python3 /home/nik/data.preprocessing.py --data='/home/nik/data/ml_airbnb_train.csv' 
# python3 /home/nik/data.preprocessing.py --data='/home/nik/regression_and_multiclass_datasets/nik_multiclass_dna_protein.csv' --id='multiclass_dna_protein'
# python3 /home/nik/data.preprocessing.py --data='/home/nik/regression_and_multiclass_datasets/nik_multiclass_glass.csv' --id='multiclass_glass'
# python3 /home/nik/data.preprocessing.py --data='/home/nik/regression_and_multiclass_datasets/nik_multiple_regression_daily_orders_forecasting.csv' --id='multiple_regression_daily_orders_forecasting'
# python3 /home/nik/data.preprocessing.py --data='/home/nik/regression_and_multiclass_datasets/nik_multiple_regression_facebook_interactions.csv' --id='multiple_regression_facebook_interactions'
# python3 /home/nik/data.preprocessing.py --data='/var/vizadata/data/ml_inno_train.csv'  --id='inno'
# python3 /home/nik/data.preprocessing.py --data='/var/vizadata/data/ml_inno_train.csv'  --id='inno_24' --min_mem_size=24
# ssh nik@35.155.140.237
# scp nik@35.155.140.237:/home/nik/*.csv  ./
# scp nik@35.155.140.237:/var/vizadata/data/ml_inno_train.csv  ./
# scp *.py nik@35.155.140.237:/home/nik/  
# ps -fA | grep python
# scp nik@35.155.140.237:/home/nik/z6AzOlBAO/*.csv  ./
# ps -fea|grep -i java
# rm -rf mydir
'''
sudo pip3 uninstall h2o
sudo pip3 install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
'''
import h2o
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
parser.add_option('-d', '--data', help='data file path')
parser.add_option('-t', '--target', help='dependent variable')
parser.add_option('-i', '--id', help='run id')
parser.add_option('-p', '--path', help='server path')
parser.add_option('-m', '--min_mem_size', help='minimum memory size')
parser.add_option('-r', '--num_rows', help='number of rows')
(opts, args) = parser.parse_args()


# Functions

def alphabet(n):
  alpha='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'    
  str=''
  r=len(alpha)-1   
  while len(str)<n:
    i=random.randint(0,r)
    str+=alpha[i]   
  return str
  
  

def dict_to_json(dct,n):
  j = json.dumps(dct, indent=4)
  f = open(n, 'w')
  print(j, file=f)
  f.close()
  


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
    
def set_params(num_rows,server_path,data_path,process_id,variables,target,min_mem_size,run_time,classification,scale,max_models,balance_y,balance_threshold,analysis,level_set,levels_thresh):
  params={}
  params['num_rows']=num_rows  
  params['server_path']=server_path
  params['data_path']=data_path
  params['process_id']=process_id
  params['variables']=variables
  params['target']=target
  params['analysis']=analysis
  params['max_models']=max_models
  params['classification']=classification
  params['scale']=scale
  params['balance']=balance_y
  params['balance_threshold']=balance_threshold
  params['min_mem_size']=min_mem_size
  params['levels_thresh']=levels_thresh
  params['level_set']=level_set
  params['start_time']= time.time()
  return params
    
    
def get_analysis_type_y(df,y,lev=33):
  analysis=0
  levels=100000
  type_y=''
  classification=False
  for key, val in df[y].types.items():
    type_y=val
  try:
    levels=len(df[y].levels()[0])
  except:
    pass 
  f=[0]
  try:    
    f = df[y].asfactor()
  except:
    pass 
  try:
    levels=len(f.levels()[0])
  except:
    levels=100000    
  if levels == 2:
    analysis=1 # 0 - none, 1- binary, 2 - multi-class, 3 - regression
    classification=True
  elif (levels < lev) and (levels > 1):
    analysis=2 # 0 - none, 1- binary, 2 - multi-class, 3 - regression  
    classification=True
  if (type_y in ['int']) and (classification==False):
    analysis=3 # 0 - none, 1- binary, 2 - multi-class, 3 - regression  
  if type_y in ['real']:
    classification=False
    analysis=3 # 0 - none, 1- binary, 2 - multi-class, 3 - regression  
  return type_y,levels,analysis,classification

# 0 - none, 1- binary, 2 - multi-class, 3 - regression

def get_variables_types(df, l):
  Xd={}
  for key, val in df.types.items():
    if key in l:
      Xd[key]=val
  return Xd


def get_variable_names(df):  
    n = [name for name in df.columns]              
    return n

def get_variables(df,dct,thresh=2):
    d={}
    for key, val in df.types.items():
        d[key]=val
        if val == 'int':
        # Check range then if can be enum with less than thresh levels
        # Also check is every field unique or the same
          try:
            if key in dct.keys():
              if dct[key][1] <= thresh:   
                d[key]='enum'    
          except:
            pass            
    return d

def save_variables(dct, path):
  (f,ext)=os.path.splitext(path) 
  n=os.path.basename(f)+'_variables'+ext 
  var_df = pd.DataFrame.from_dict(dct, orient="index")  
  var_df = var_df.transpose()   
  var_df.to_csv(n, index=False)
  return n


#  End Functions


min_mem_size=4
pct_memory=0.9
virtual_memory=psutil.virtual_memory()
max_mem_size=int(round(int(pct_memory*virtual_memory.available)/1073741824,0))
if min_mem_size > max_mem_size:
  min_mem_size=max_mem_size-1


#try:

# if True:

try:

  data_path=None
  if opts.data is not None:
    try:
      data_path=str(opts.data)  # data_path = '/Users/bear/Downloads/H2O/VD/data/loan.csv'
    except:
      sys.exit(5)
      
  target=None
       
  if opts.target is not None:
    try:
      target=str(opts.target)  # target = 'dependent_variable'
    except:
      sys.exit(5)
      
          
  
  process_id=None
  
  if opts.id is not None:
    try:
      process_id=str(opts.id)  # 
    except:
      sys.exit(5) 
      
      
      
  server_path=None
  
  if opts.path is not None:
    try:
      server_path=str(opts.path)  # server_path='/Users/bear/Downloads/H2O/VD'
    except:
      sys.exit(5) 
      
  min_mem_size=6 
  
  if opts.min_mem_size is not None:
    try:
      min_mem_size=int(opts.min_mem_size)  
    except:
      sys.exit(5)       

  num_rows=100000
       
  if opts.num_rows is not None:
    try:
      num_rows=int(opts.num_rows)  # set num_rows
    except:
      sys.exit(5)   


  variables=None
  run_time=360
  classification=False
  scale=False
  max_models=9    
  balance_y=False 
  balance_threshold=0.2
  levels_thresh=33
  level_set=[]
  analysis=1 # 0 - none, 1- binary, 2 - multi-class, 3 - regression
  
  # import h2o

  # h2o.cluster().shutdown()
  # 65535 Highest port no
  port_no=random.randint(5555,55555)  
  

#  h2o.init(strict_version_check=False,min_mem_size_GB=min_mem_size,port=port_no) # start h2o
  try:
    h2o.init(strict_version_check=False,min_mem_size_GB=min_mem_size,max_mem_size_GB=max_mem_size,port=port_no) # start h2o
  except:
    logging.critical('h2o.init')
    h2o.download_all_logs(dirname=logs_path, filename=logfile)     
    h2o.cluster().shutdown()
    sys.exit(2)
  
  # Init params
  
  params=set_params(num_rows,server_path,data_path,process_id,variables,target,min_mem_size,run_time,classification,scale,max_models,balance_y,balance_threshold,analysis,level_set,levels_thresh)
  
  
  if process_id==None:
    process_id=alphabet(9)
  if server_path==None:
    server_path=os.path.abspath(os.curdir)
  os.chdir(server_path) 
  
  run_dir = os.path.join(server_path,process_id)
  os.mkdir(run_dir)
 # os.chdir(run_dir)    
  
  print (process_id) # run_id to std out
  
  
  # Automodeler start
  # Logs
  
  logfile=process_id+'_autoh2o_log.zip'
  logs_path=os.path.join(run_dir,'logs')  
  
  # logging
  log_file=process_id+'.log'
  log_file = os.path.join(server_path,log_file)
  logging.basicConfig(filename=log_file,level=logging.INFO,format="%(asctime)s:%(levelname)s:%(message)s")
  logging.info(datetime.now()) 
    
  
  try:
    df = h2o.import_file(data_path)
  except:
    logging.critical('h2o.import_file(data_path)')   
    h2o.download_all_logs(dirname=logs_path, filename=logfile)       
    h2o.cluster().shutdown()
    sys.exit(3)
  

  # Cut data to roughly num_rows if data have more rows than num_rows
  pct_rows=round(num_rows/df.shape[0],2)
  if pct_rows < 0.02:
    pct_rows=0.02 
  train, test, valid = [[],[],[]]
  half_pct=0.0
  if pct_rows < 1:
    half_pct=round(((1-pct_rows)/2),2)  
    df, test, valid = df.split_frame([pct_rows,half_pct])
  
  params['pct_data']=[pct_rows,half_pct]  
  
  # Get head 
  head=df.head()
  head=head.as_data_frame()  
  
  head.to_csv(process_id+'_head.csv')
  
  
  # get variable counts
  v=get_variable_names(df)

  unique_counts={}
  
  scale=1
  if pct_rows < 1:
    scale=(1/pct_rows)    
    
    
  for n in v:
    try:
      l=[]
      l.append(int(df[n].nrows*scale))      
      try:
        nunique=int(len(df[n].unique()))
      except:
        nunique=int(df[n].nrows*scale)    
      l.append(nunique) 
      l.append(int(df[n].nacnt()[0]*scale))    
      unique_counts[n]=l
    except:
      pass      

        
  params['count_nunique_isnull']=unique_counts 
  
  # Output data summary stats
    
#  describe=df_pandas.describe()  
#  describe.to_csv(process_id+'_describe_head.csv')
  

  
  # dependent variable
  # assign target and inputs for classification or regression
  if target==None:
    target=df.columns[-1]   
  y = target
  
  yd=get_variables_types(df, y)
  
  params['target']=yd 
  
  
  # Type of analysis
  
  type_y,levels,analysis,classification=get_analysis_type_y(df,y,levels_thresh)
  
  # level_set
  
  if classification:
    df[y] = df[y].asfactor()
    level_set=df[y].levels()[0] 
  
  # Update params
  
  params['process_id']=process_id
  params['analysis']=analysis 
  params['classification']=classification
  params['level_set']=level_set  
  
  # variables
    
  
  Xd = get_variables(df, unique_counts, 2) 
   
  
  params['variables']=Xd 
  
  # Save variables to csv
  n=save_variables(Xd, data_path)
  params['variables_file'] = n  
  
  # Time script
  
  params['end_time'] = time.time()
  params['execution_time'] = params['end_time'] - params['start_time']
  
  
  # Save params
  
  n=process_id+'_data_preprocessing_stats.json'
  dict_to_json(params,n)
  
  # Save logs
  h2o.download_all_logs(dirname=logs_path, filename=logfile)  
  
  # Clean up
  
  os.chdir(server_path)
  
  h2o.remove_all()  

  h2o.cluster().shutdown()
      
  sys.exit(0)
 

except Exception as e:
  print(e)  
  logging.critical('h2o.unknown')
  h2o.download_all_logs(dirname=logs_path, filename=logfile)   
  h2o.cluster().shutdown()
  sys.exit(9)
  
  

