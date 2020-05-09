# python3 data_scrubber_edited_withfunctions_copy.py --data='/home/ec2-user/Prabhu/word2vec/DAT/AmazonReviews.Small.csv' --prod='/home/ec2-user/Prabhu/word2vec/DAT/AmazonReviews.Small.csv'


import h2o
from h2o.automl import H2OAutoML
import random, os, sys
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import csv
import optparse
import time
import json
from distutils.util import strtobool
import psutil
from h2o.estimators.word2vec import H2OWord2vecEstimator
import statistics


# data_path=None
# prod_path=None
# stopword_path=None
# min_mem_size=6 
# server_path=None 
# rid=None
# data_orig=None
# prod_orig=None
# balance=False # Set balance=False to turn off balance  
# time_factors=False # Set time_factors=False to turn off time factors
# scrubber=True # Set scrubber=False to turn off scrubber
# word2vec=True # Set text_field=False to turn off text extraction using word2vec
# bad_levels=False  # Set bad_levels=False to production level remover
# features=False  # Set features=False to production feature selection
# classification=False
# no_rows=100000
# target='is_buyer'
# target='label'
# target='Knee Issue'
# target='Survived' # Required for feature selection
# all_variables=None
# run_time=111



parser=optparse.OptionParser()
parser.add_option('-b', '--bad_levels', help='set bad levels=to false to turn off')
parser.add_option('-c', '--classification', help='classification yes/no')
parser.add_option('-d', '--data', help='data file path')
parser.add_option('-p', '--path', help='server path')
parser.add_option('-q', '--prod', help='production data')

parser.add_option('-l', '--avg_char_len', help='Average Character threshold for word2vec')
parser.add_option('-z', '--n_vec', help='Number of Vectors for word2vec')
parser.add_option('-e', '--min_mem_size', help='minimum memory size')
parser.add_option('-f', '--time_factors', help='time factors')
parser.add_option('-g', '--balance', help='balance')  
parser.add_option('-i', '--all_variables', help='all variables')
parser.add_option('-j', '--features', help='features')

parser.add_option('-r', '--run_time', help='run time')
parser.add_option('-s', '--scrubber', help='set scrubber to false to turn off')
parser.add_option('-w', '--word2vec', help='set word2vec to false to turn off')
parser.add_option('-t', '--target', help='dependent variable')
parser.add_option('-x', '--rid', help='run id')
(opts, args) = parser.parse_args()


# if True:
try:
    # parser.add_option('-d', '--data', help='data file path')
    data_path=None
    if opts.data is not None:
        try:
            data_path=str(opts.data)  # data_path = '/Users/bear/Downloads/H2O/VD/data/loan.csv'
        except:
            sys.exit(5) 
    
    # parser.add_option('-q', '--prod', help='production data')
    prod_path=None
    if opts.prod is not None:
        try:
            prod_path=str(opts.prod)  # prod = '/Users/bear/Downloads/H2O/VD/data/loan.prod.csv'
        except:
            sys.exit(5)   
    
    # parser.add_option('-x', '--rid', help='run id')   
    rid=None
    if opts.rid is not None:
        try:
            rid=str(opts.rid)  # prod = '/Users/bear/Downloads/H2O/VD/data/loan.prod.csv'
        except:
            sys.exit(5) 
    
    # parser.add_option('-l', '--char_len', help='Character threshold for word2vec')
    avg_char_len=100
    if opts.avg_char_len is not None:
        try:
            avg_char_len=int(opts.avg_char_len)
        except:
            sys.exit(5)
            
    # parser.add_option('-z', '--vec_size', help='Vector size for word2vec')
    n_vec=50
    if opts.n_vec is not None:
        try:
            n_vec=int(opts.n_vec)
        except:
            sys.exit(5)
            
    # parser.add_option('-e', '--min_mem_size', help='minimum memory size')
    if opts.min_mem_size is not None:
        try:
            min_mem_size=int(opts.min_mem_size)  
        except:
            sys.exit(5)     
    
    classification=False
        
    if opts.classification is not None:
        try:
            classification=bool(strtobool(opts.classification))  # classification=True
        except:
            sys.exit(5)     
        
    bad_levels=True  # Set bad_levels=False to production level remover    
    if opts.bad_levels is not None:
        try:
            bad_levels=bool(strtobool(opts.bad_levels))  
        except:
            sys.exit(5)  
        
    # run_time=360       
    run_time=360
    
    if opts.run_time is not None:
        try:
            run_time=int(opts.run_time)  # run_time=360
        except:
            sys.exit(5)      
    
    features=True # Set time_factors=False to turn off time_factors    
    if opts.features is not None:
        try:
            features=bool(strtobool(opts.features))  
        except:
            sys.exit(5)      
    
    time_factors=True # Set time_factors=False to turn off time_factors    
    if opts.time_factors is not None:
        try:
            time_factors=bool(strtobool(opts.time_factors))  
        except:
            sys.exit(5)  
                    
    balance=True # Set balance=False to turn off balance    
    if opts.balance is not None:
        try:
            balance=bool(strtobool(opts.balance))  
        except:
            sys.exit(5)   
                            
    scrubber=True # Set scrubber=False to turn off scrubber     
     # scrubber=False # Set scrubber=False to turn off scrubber   
    if opts.scrubber is not None:
        try:
            scrubber=bool(strtobool(opts.scrubber))  
        except:
            sys.exit(5)  
    
    word2vec=True # Set word2vec=False to turn off word2vec     
    # word2vec=False # Set word2vec=False to turn off word2vec   
    if opts.word2vec is not None:
        try:
            word2vec=bool(strtobool(opts.word2vec))  
        except:
            sys.exit(5)  
    
    server_path=None
    if opts.path is not None:
        try:
            server_path=str(opts.path)  # server_path='/Users/bear/Downloads/H2O/VD'
        except:
            sys.exit(5) 
    
    all_variables=None
    if opts.all_variables is not None:
        try:
            all_variables=str(opts.all_variables)  # data_path = '/Users/bear/Downloads/H2O/VD/data/loan.fields.csv'
        except:
            sys.exit(5) 
        
    target=None
    if opts.target is not None:
        try:
            target=str(opts.target)  # target = 'dependent_variable'
        except:
            sys.exit(5)            
    # End of parser
    
    
    # Functions
    def alphabet(n):
        alpha='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'    
        str=''
        r=len(alpha)-1   
        while len(str)<n:
            i=random.randint(0,r)
            str+=alpha[i]   
        return str
         
        
    def set_meta_data(run_id,server,data,path,prod,min_mem_size):
        m_data={}
        m_data['start_time'] = time.time()
        m_data['server_path']=server
        m_data['data_path']=data 
        m_data['prod_path']=prod
        m_data['run_id'] =run_id
        m_data['end_time'] = time.time()
        m_data['execution_time'] = 0.0
        m_data['run_path'] =path
        m_data['min_mem_size'] = min_mem_size
        return m_data  
    
    def dict_to_json(dct,n):
        j = json.dumps(dct, indent=4)
        f = open(n, 'w')
        print(j, file=f)
        f.close()
    
    def get_levels_diff(dat, pro):
        bad_levels={}
        good_levels={}    
        cnt=0    
        ints, reals, enums = [], [], []  
        lst=list(set(dat.columns).intersection(pro.columns))
        pro_types=dict(pro.types.items())      
        for key, val in df.types.items():
            if key in lst:
                if val == 'enum':
                    l=[]
                    try:
                        l=dat[key].levels()[0] 
                    except:
                        pass            
                    p=[]  
                    if pro_types[key] == 'enum':
                        try:
                            p=pro[key].levels()[0] 
                        except:
                            pass 
                    b=list(set(p) - set(l)) 
                    if len(b)>0:
                        bad_levels[key]=b 
                        good_levels[key]=l                 
                        cnt+=len(b)
        return bad_levels,good_levels,cnt,lst  
    
    def get_scrubbed_name(p):
        n=''
        (data_root, data_ext) = os.path.splitext(p)
        data_names=data_root.split('/')
        n=data_names[-1]+'_scrubbed'+data_ext
        return n  
      
    def split_date_time_to_columns(df):
        try:
            for key,value in df.types.items():
                if(value=='time'):
                    df[key + '_day'] = df[key].day()
                    df[key + '_month'] = df[key].month()
                    df[key + '_week'] = df[key].week()
                    df[key + '_year'] = df[key].year()
                    df[key + '_dayOfWeek'] = df[key].dayOfWeek()
                    df[key + '_hour'] = df[key].hour()
                    df[key + '_minute'] = df[key].minute()
                    df[key + '_second'] = df[key].second()
            return df
        except:
            pass
        
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
    
    def get_model_by_algo(algo,models_dict):
        mod=None
        mod_id=None    
        for m in list(models_dict.keys()):
            if m[0:3]==algo:
                mod_id=m
                mod=h2o.get_model(m)
                return mod,mod_id
        return mod,mod_id 
    
    def feature_selection(d,l, thresh=0.05):
        iv=[]
        for key in d.keys():
            if d[key]>thresh:
                iv.append(key)
        for v in iv:
            if v not in l:
                iv.remove(v)
        return iv
    
    
    
    #Calculating Ratio of the imbalanced dataset
    def find_imbalanced_ratio(data, target, threshold=0.4): 
        maxPercentage = (data[target].value_counts().max()/data[target].count())*100
        minPercentage = (data[target].value_counts().min()/data[target].count())*100
        if minPercentage < threshold * 100 :
            print("Dataset is Imbalanced")
        else:
            print("Dataset is Balanced")
        return maxPercentage, minPercentage
    
    def make_over_samples_SMOTE(data,target, threshold=0.4):
     #input DataFrame
     #X →Independent Variable in DataFrame\
     #y →dependent Variable in Pandas DataFrame format
        Xt = data.iloc[:, data.columns != target]
        yt = data.iloc[:, data.columns == target]
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(ratio = threshold)
        Xt, yt = sm.fit_sample(Xt, yt)
        return Xt,yt    
    
    # word2vec training and getting a dataset as a final with all the vectors binded with the original dataset
    def word2vec_function(df, char_len, vec_size):
        char_l=df.ascharacter().nchar()
        charlen_arr=char_l.mean()
        count=0
        arr=[]
        for mean in charlen_arr:
            if round(mean)>=avg_char_len:
                arr.append(count)
            count+=1
        for val in arr:
            words = tokenize(df[val].ascharacter())
            col=df.columns[val]
            model_id= "w2v_"+col+".hex"
            w2v_model = H2OWord2vecEstimator(vec_size = n_vec, model_id = model_id)
            w2v_model.train(training_frame=words)
            df_vec = w2v_model.transform(words, aggregate_method = "AVERAGE")
            df_vec.names = [col + '_W2V_{}'.format(i+1) for i in range(len(df_vec.names))]
            ext_df = df.cbind(df_vec)
            df = ext_df
        return df
    
    
    # Stopwords for the testings is included
    # Below code tests if Stopping word is present in the local machine, if not then it finds it at the specific GitHub 
    stopword_file = "data/stopwords.csv"
    if os.path.isfile(stopword_file):
        stopword_file = stopword_file
    else:
        stopword_file = "https://raw.githubusercontent.com/h2oai/h2o-tutorials/master/h2o-world-2017/nlp/stopwords.csv"
    
    STOP_WORDS = pd.read_csv(stopword_file, header=0)
    STOP_WORDS = list(STOP_WORDS['STOP_WORD'])
    
    # Function to tockenize the words compared to the stop words
    def tokenize(sentences, stop_word = STOP_WORDS):
        tokenized = sentences.tokenize("\\W+")
        tokenized_lower = tokenized.tolower()
        tokenized_filtered = tokenized_lower[(tokenized_lower.nchar() >= 2) | (tokenized_lower.isna()),:]
        tokenized_words = tokenized_filtered[tokenized_filtered.grep("[0-9]",invert=True,output_logical=True),:]
        tokenized_words = tokenized_words[(tokenized_words.isna()) | (~ tokenized_words.isin(STOP_WORDS)),:]
        return tokenized_words
    
    def clean_html(df):
        arr=df.columns
        for col in arr:
            print(col)
            if str(df[col].types.values())=="dict_values(['string'])" or str(df[col].types.values())=="dict_values(['enum'])":
                print('Removing the HTML tags: {}'.format(col))
                try:
                    df[col]=df[col].gsub("<br />", ' ', ignore_case=True)
                    df[col]=df[col].gsub("\n", ' ', ignore_case=True)
                    df[col]=df[col].gsub("<", ' ', ignore_case=True)
                    df[col]=df[col].gsub("/>", ' ', ignore_case=True)
                    df[col]=df[col].gsub(", ", '', ignore_case=True)
                    df[col]=df[col].gsub("/a>", ' ', ignore_case=True)
                    df[col]=df[col].gsub('"' , '', ignore_case=True)
                except:
                    pass
        return df
        
        
    def get_sample(df, row_count):
        no_rows=round(df.shape[0]*0.9,0)
        print(no_rows)
        if no_rows> row_count:
            no_rows=row_count
        print(no_rows)
        # Cut data to roughly no_rows if data have more rows than no_rows
        print(no_rows/df.shape[0])
        pct_rows=round(no_rows/df.shape[0],2)
        if pct_rows < 0.02:
            pct_rows=0.02   
        print(pct_rows)
        
        train, test, valid = [[],[],[]]
        half_pct=0.0
        if pct_rows < 1:
            half_pct=round(((1-pct_rows)/2),2)  
            df, test, valid = df.split_frame([pct_rows,half_pct])
        return df, test, valid 
    #  End Functions
    
    
    # server_path='/home/ec2-user/Prabhu/word2vec/DAT'
    # data_file='DAT/digiq_wine_multiclass_training.csv'
    # data_file='DAT/voice.csv'
    # data_file='DAT/SAMPLE_ML_TRAINING_OR_KNEE_2011_2015_DATA+-+NLC+Mutual.csv'
    # data_file='DAT/titanic_train.csv'
    # data_file='data/AmazonReviews.Small.csv'
    # data_file='DAT/yelp.csv'
    # stopword_file='data/stopwords.csv'
    # data_path=os.path.join(server_path,data_file)
    # data_path
    
    
    # prod_file='DAT/digiq_wine_multiclass_production.csv'
    # prod_file='DAT/voice.csv'
    # prod_file='DAT/SAMPLE_ML_PRODUCTION_OR_KNEE_2016_DATA+-+NLC+Mutual.csv'
    # prod_file='DAT/AmazonReviews.Small.csv'
    # prod_file='DAT/yelp.csv'
    # prod_path=os.path.join(server_path,prod_file)
    # prod_path
    
    
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
    
    
    # 65535 Highest port no
    port_no=random.randint(5555,55555)
    
    min_mem_size=6
    pct_memory=0.9
    virtual_memory=psutil.virtual_memory()
    max_mem_size=int(round(int(pct_memory*virtual_memory.available)/1073741824,0))
    if min_mem_size > max_mem_size:
        min_mem_size=max_mem_size
    
    #  h2o.init(strict_version_check=False,min_mem_size_GB=min_mem_size,port=port_no) # start h2o
    try:
        h2o.init(strict_version_check=False,min_mem_size_GB=min_mem_size,port=port_no) # start h2o
#         print("h2o started")
    except:
        logging.critical('h2o.init')
        h2o.download_all_logs(dirname=logs_path, filename=logfile)      
        h2o.cluster().shutdown()
        sys.exit(2)
    
    
    # meta data
    # set_meta_data(run_id,server,data,path,prod,min_mem_size) 
    meta_data = set_meta_data(run_id,server_path,data_path,prod_path,run_dir,min_mem_size)
#     print(meta_data)
    
#     data_path="/home/ec2-user/Prabhu/word2vec/DAT/titanic_train.csv"
    print(data_path)
    
    try:
        df = h2o.import_file(data_path)
    except:
        logging.critical('h2o.import_file(data_path)')    
        h2o.download_all_logs(dirname=logs_path, filename=logfile)    
        h2o.cluster().shutdown()
        sys.exit(3)
    
    print(df)
    # Load and analyze production versus training files
    data_orig=df
    
    prod=None  
    
    if prod_path is not None:
        try:  
            prod = h2o.import_file(prod_path,header=1)  
        except:
            logging.critical('h2o.import_file(prod_path)')    
            h2o.download_all_logs(dirname=logs_path, filename=logfile)    
            h2o.cluster().shutdown()
            sys.exit(3)
    
    
    prod=clean_html(prod)
    df=clean_html(df)
    
    
    if prod is not None:
        prod_orig=prod
    
    
    df_pd=None
    
    
    samples_df, test, valid=get_sample(df, 10000)
    
    
    samples_df
    
    
    samples_df=clean_html(samples_df)
    
    
    if word2vec is True:
        print("Am performing word2vec")
        word2vec_df=word2vec_function(samples_df, avg_char_len, n_vec)# samples_df=df
        print(word2vec_df)
        
    df=word2vec_df
    
    # Set balance=False to turn off balance 
    if (balance is True) and (target is not None):
        df_pd=df.as_data_frame()
        maxPercentSample, minPercentSample = find_imbalanced_ratio(df_pd, target)
        meta_data['max_min_percent']=[maxPercentSample,minPercentSample]
        Xi,yi = make_over_samples_SMOTE(df_pd,target)
        result = np.column_stack([Xi, yi])
        #saving the column names from data in a list
        cols = list(df_pd.columns.values)
        # printing column names from dataframe
        df2 = pd.DataFrame(result)
        #assigning columns names to new dataframe df2
        df2.columns = cols
        df = h2o.H2OFrame(df2)
    
    
    # Set time_factors=False to turn off time factors
    if (time_factors is True):
        df=split_date_time_to_columns(df)
        if prod is not None:
            prod=split_date_time_to_columns(prod)
    
    
    # Set up AutoML
    aml = H2OAutoML(max_runtime_secs=run_time)
    
    
    # Start feature selection
    alg_start_time = time.time()
    if (features is True) and (target is not None):
        no_rows=round(df.shape[0]*0.9,0)  # 90% of rows
        if no_rows> 100000:  # If 90% > 100000 limit to 100000 
            no_rows=100000
    
    # Cut data to roughly no_rows if data have more rows than no_rows
        pct_rows=round(no_rows/df.shape[0],2)
        if pct_rows < 0.02:
            pct_rows=0.02   
        
        train, test, valid = [[],[],[]]
        half_pct=0.0
        if pct_rows < 1:
            half_pct=round(((1-pct_rows)/2),2)  
            train, test, valid = df.split_frame([pct_rows,half_pct])
        
        if target==None:
            target=train.columns[-1] 
        
        y = targe
        X = []  
        if all_variables is None:
            X=get_independent_variables(train, target)  
        else: 
            ivd=get_all_variables_csv(all_variables)    
            X=check_all_variables(train,ivd,y)
        
        if classification:
            train[y] =  train[y].asfactor()
        
        try:
            aml.train(x=X,y=y,training_frame=train)  # Change training_frame=train
        except Exception as e:
            logging.critical('aml.train') 
            h2o.download_all_logs(dirname=logs_path, filename=logfile)      
            h2o.cluster().shutdown()   
            sys.exit(4)    
        
        aml_leaderboard_df=aml.leaderboard.as_data_frame()    
         
        models_dict={}
        for m in aml_leaderboard_df['model_id']:
            models_dict[m]=None 
        
        varimp={}
          
        mod,mod_id=get_model_by_algo("DRF",models_dict)
        if mod is not None:
            l=mod.varimp()
            for v in l:
                varimp[v[0]]=v[2]
        
        mod,mod_id=get_model_by_algo("XRT",models_dict)
        if mod is not None:    
            l=mod.varimp()
            for v in l:
                if v[0] in varimp:   
                    varimp[v[0]]=((v[2]+varimp[v[0]])/2)
                else:   
                    varimp[v[0]]=((v[2]+0.0)/2)
        
        mod,mod_id=get_model_by_algo("GBM",models_dict)
        if mod is not None:    
            l=mod.varimp()
            for v in l:
                if v[0] in varimp:   
                    varimp[v[0]]=((v[2]+varimp[v[0]])/2)
                else:   
                    varimp[v[0]]=((v[2]+0.0)/2)        
        print(varimp)
        X=feature_selection(varimp,X)  
        print(X)  
        Xy=[y]    
        for e in X:
            Xy.append(e)
        print(Xy)    
        df=df[:,Xy] 
        meta_data['Xy']=Xy  
        meta_data['X']=X
        
    meta_data['feature_selection_execution_time'] = time.time() - alg_start_time     
    # End feature selection    
    
    
    bad_lev,good_lev,diff={},{},{}
    if (df is not None) and (prod is not None):
        bad_lev,good_lev,diff,common=get_levels_diff(df, prod)
    
    
    # bad_lev,good_lev,diff,common=get_levels_diff(df, prod)
    
    meta_data['num_bad_levels']=len(bad_lev.keys())
    meta_data['bad_levels']=bad_lev
    before_after=[]
    if (prod is not None):
        before_after=[prod.isna().sum(),0]
    
    
    
    if (prod is not None):
        prod.describe()
    
    nvalue = float('NaN')
    
    
    if (prod is not None) and (len(bad_lev.keys())>0) and (bad_levels is True):
        for lev in bad_lev:
            prod[lev] = (prod[lev].isin(bad_lev[lev])).ifelse('SU', prod[lev])
    
    
    # len(prod['default_city'].levels()[0])
    
    
    if (prod is not None):
        prod.describe()
    
    
    if (prod is not None):
        before_after[1]=prod.isna().sum()
    
    
    meta_data['before_after']=before_after
    
    
    cur_path=os.path.abspath(os.curdir)
    n=get_scrubbed_name(data_path)
    out_path=os.path.join(cur_path,n)
    print(n)
    print(out_path)
    meta_data['training_scrubbed_name'] = n
    
    
    '''
    if (scrubber is True):
      h2o.export_file(df, out_path)              # Scrubbed file
    else:
      h2o.export_file(data_orig,out_path) 
    '''  
    
    
    data_orig
    
    
    scrubber
    
    
    if scrubber is True:
    #   df_pd=df.as_data_frame() 
    #   df_pd.to_csv(out_path)
        h2o.export_file(df, path=out_path)
    else:
    #   data_orig=data_orig.as_data_frame()    
    #   data_orig.to_csv(out_path)
        h2o.export_file(data_orig, path=out_path)
    
    
    '''
    if (prod is not None):
      n=get_scrubbed_name(prod_path)
      print(n)
      out_path=os.path.join(cur_path,n)
      meta_data['production_scrubbed_name'] = n
      if (scrubber is True):
        h2o.export_file(prod, out_path)              # Scrubbed file
      else:
        h2o.export_file(prod_orig, out_path)   
    '''    
    
    
    # if (prod is not None):
    #   n=get_scrubbed_name(prod_path)
    #   out_path=os.path.join(cur_path,n)
    #   meta_data['production_scrubbed_name'] = n
    #   if (scrubber is True):
    #     prod_pd=prod.as_data_frame() 
    #     prod_pd.to_csv(out_path)             # Scrubbed file
    #   else:
    #     prod_pd=prod_orig.as_data_frame() 
    #     prod_pd.to_csv(out_path)             # Scrubbed file  
    
    
    # Update and save meta data
    meta_data['end_time'] = time.time()
    meta_data['execution_time'] = meta_data['end_time'] - meta_data['start_time']
    n=run_id+'_meta_data.json'
    dict_to_json(meta_data,n)    
    
    
    # Save logs
    h2o.download_all_logs(dirname=logs_path, filename=logfile)
    
    
    os.chdir(server_path)
    
    
    h2o.cluster().shutdown()
        
except Exception as e:
    print(e)  
    logging.critical('h2o.unknown')
    h2o.download_all_logs(dirname=logs_path, filename=logfile)   
    h2o.cluster().shutdown()
    sys.exit(9)    