"""
@author: Nikunj Lad

Date: 28/06/19
"""

import warnings, random, logging
from DataIO import *
from H2O import *


def get_loggers():
    logger = logging.getLogger("squark")  # name the logger as squark
    logger.setLevel(logging.INFO)
    f_hand = logging.FileHandler('preprocessor.log')  # file where the custom logs needs to be handled
    f_hand.setLevel(logging.INFO)  # level to set for logging the errors
    f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                                 datefmt='%d-%b-%y %H:%M:%S')  # format in which the logs needs to be written
    f_hand.setFormatter(f_format)  # setting the format of the logs
    logger.addHandler(f_hand)  # setting the logging handler with the above formatter specification

    return logger


def main():

    # start time just for analysis
    start = time.time()

    # # multi-processing
    # pool = mp.Pool(processes=mp.cpu_count())

    # 1. declaring variables and data structures
    latency = dict()  # latency dictionary to hold execution times of individual functions

    # 2. ignore warnings
    warnings.filterwarnings("ignore")  # ignoring warnings and other errors

    # 3. get custom logger
    code_dir = os.path.dirname(os.path.abspath(__file__))

    if os.path.exists(code_dir + '/preprocessor.log'):
        os.remove(code_dir + '/preprocessor.log')

    logger = get_loggers()
    logger.info("Logs initiated!")

    # 3. parsing the configuration files which are defined for user tunable parameters
    parser = ConfigParser()  # defining the parser
    parser.read("config.ini")  # reading the config file

    # 4. getting parameters dictionary which will act as metadata for this file
    io = DataIO(latency, logger)  # defining the data io object
    df_obj = io.parse_json("default_config.json")  # parse the json
    args = io.parse_configurations(parser, df_obj, 'preprocessor')  # get the meta data as well as the latency info
    # args = pool.apply(io.parse_configurations(), args=(parser, df_obj, 'preprocessor'))
    args['start_time'] = time.time()

    # initialization of h2o
    latency = io.latency
    port_no = random.randint(5555, 55555)

    # 6. if run_id is none then generate new one using the UUID library
    if args['run_id'] is None:
        args['run_id'] = io.generate_uuid(8)
        temp = io.generate_uuid(5)
        # args['run_id'] = pool.apply_async(io.generate_uuid(), args=(8))

    # 7. if server_path is none then append it with the current directory path
    if args['server_path'] is None:
        args['server_path'] = os.path.abspath(os.curdir)

    # 8. change directory to the server path
    os.chdir(args['server_path'])
    args['code_dir'] = code_dir
    args['run_dir'] = os.path.join(args['server_path'],
                                   args['run_id'])  # run_dir is the server path plus the uuid directory
    os.mkdir(args['run_dir'])  # change to the new server path and create a new directory
    print("Run id: ", args['run_id'])  # print this for information

    # 9. h2o logs path
    h2o_logs_file = args['run_id'] + '_autoh2o_log.zip'  # h2o log file
    h2o_logs_path = os.path.join(args['run_dir'], 'logs')  # h2o logs path

    # 10. initialization of h2o using
    ho = H2O(latency=latency, port=port_no, h2o_logs_file=h2o_logs_file, h2o_logs_path=h2o_logs_path,
             server_path=args['server_path'], logger=logger)
    min_mem_size, max_mem_size = ho.cluster_min_mem()
    ho.init_h2o(min_mem_size=min_mem_size, max_mem_size=max_mem_size, strict_version_check=False)

    # 11. import data from external file
    df = ho.h2o_import_data(args['data_path'])

    # 12. train test splitting of data based on percentage
    pct_rows, half_pct, df, test, valid = ho.split_h2o_dataframe(args['num_rows'], df.shape[0],
                                                                 args['fraction_of_rows'], df)
    args['percent_rows'] = [pct_rows, half_pct]

    # 13. get h2o dataframe head
    head = df.head()
    head = head.as_data_frame()

    # 14. export data to csv
    csv_name = args['run_id'] + '_head.csv'
    io.export_to_csv(head, args['run_dir'] + '/' + csv_name)

    # 15. get just the list of h2o dataframe column names
    var_names = ho.get_variable_names(df)

    # 16. dictionary to hold stats of total values, unique values and null values in each column
    unique_counts = ho.get_unique_counts(pct_rows, var_names, df)
    args['count_nunique_isnull'] = unique_counts

    # 17. last column selection if target is None
    if args['target_var'] is None:
        args['target_var'] = df.columns[-1]
    y = args['target_var']

    # 18. get the types of the variables given the dataframe
    yd = ho.get_variables_types(df, y)
    args['target'] = yd

    # 19. get the type of analysis needs to be done, type of target variable,
    type_y, levels, analysis, classification = ho.get_analysis_type(df, y, args['levels_thresh'])

    # 20. checking if classification is the type of analysis to be done on data
    if classification:
        df[y] = df[y].asfactor()
        args['level_set'] = df[y].levels()[0]

    args['analysis'] = analysis
    args['classification'] = classification

    # 21. get the variables
    Xd = ho.get_variables(df, unique_counts, 2)

    # 22. adding more parameters to our variables dictionary
    args['variables'] = Xd
    args['end_time'] = time.time()
    args['execution_time'] = args['end_time'] - args['start_time']

    # 23. save parameters to JSON file
    file_name = args['run_dir'] + '/' + args['run_id'] + '_data_preprocessing_stats.json'
    io.export_to_json(args, file_name)

    if os.path.isfile(args['code_dir'] + '/preprocessor.log'):
        os.rename(args['code_dir'] + '/preprocessor.log', args['run_dir'] + '/preprocessor.log')

    # export latency information
    file_name = args['run_dir'] + '/' + args['run_id'] + '_latency_stats.json'
    io.export_to_json(io.latency, file_name)

    stop = time.time()
    logger.info('Execution time : ' + str(stop - start))

    # 24. shutdown and terminate everything
    ho.h2o_shutdown()


if __name__ == '__main__':
    main()
