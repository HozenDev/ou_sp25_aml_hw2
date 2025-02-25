#################################################################
def aggregate_stats(args:argparse.ArgumentParser, results_keys:[str]=[], exp_keys:[str]=[]):
    '''
    Aggregate all of the run/results data together for a single set of runs.

    Uses the JobIterator as configured by args.exp_type to enumerate all of the
    result pickle files.  These are loaded sequentially, and the details of each
    run (configuration and results) are then placed into a single dataframe.

    :param args: The arguments from the command line
    :param results_keys: List of strings that are the keys to extract from
                 the results dictionary and store in the dataframe.  Note that
                 these keys must access scalar values!
    :param exp_keys: List of strings that are the keys to extract from
                 the args object and store in the dataframe.  Note that these
                 keys must access scalar values!

    :return: A dataframe that contains one row for each experiment (except for
                 those that have missing pkl file)
    '''

    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be loading
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    # output dataframe list
    dfs = []

    # Iterate over every experiment in the Cartesian product
    for i in range(ji.get_njobs()):
        # Dictionary that describes the DataFrame row
        d = {}

        # Set the exp index to this iteration
        args.exp_index = i

        # Use the exp_index and exp_type to override
        #  specified arguments.  It is a little weird to
        #  do it this way (having two identical JobIterators),
        #  but it does work
        params_str = augment_args(args)

        # Log the experiment conditions
        for k in exp_keys:
            d[k] = getattr(args, k, None)

        # Compute input file name base
        fbase = generate_fname(args, params_str)
    
        # Full input pickle file name
        fname_in = "%s_results.pkl"%(fbase)

        print("File name:", fname_in)

        d['fbase'] = fbase
        d['fname'] = fname_in

        # If the file exists, then load it and create the row
        if os.path.exists(fname_in):
            with open(fname_in, "rb") as fp:
                # Load the data
                results = pickle.load(fp)

                # Log the performance metrics
                for k in results_keys:
                    d[k] = results[k]

                # New row to the dataframe
                dfs.append(pd.DataFrame([d]))
        else:
            # TODO: should we include a blank row?
            print("WARNING -- file does not exist: %s"%fname_in)

    # Combine all of the rows together into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    return df


#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    # Turn off GPUs?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')


    # GPU check
    visible_devices = tf.config.get_visible_devices('GPU')
    n_visible_devices = len(visible_devices)

    print('GPUS:', visible_devices)
    if(n_visible_devices > 0):
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_visible_devices)
    else:
        print('NO GPU')

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

    # Which job to do?
    if(args.check):
        # Just look at which results files have NOT been created yet
        check_completeness(args)
    elif(args.meta):
        # All of the results to pull out of the results pickle file.  Note: these must be scalars!
        results_keys = ['predict_testing_eval_loss', 'predict_testing_eval_fvaf', 'predict_testing_eval_rmse',
                        'predict_validation_eval_loss', 'predict_validation_eval_fvaf', 'predict_validation_eval_rmse',
                        'predict_training_eval_loss', 'predict_training_eval_fvaf', 'predict_training_eval_rmse']

        # All of the experiment configuration data to pull out of the pickle file.  These must be scalars, too!
        exp_keys = ['exp_type', 'L1_regularization', 'L2_regularization', 'label', 'Ntraining', 'output_type',
                    'predict_dim', 'dropout', 'exp_index', 'rotation']

        # Use jobiterator to load each of the corresponding pkl files and construct a dataframe
        df = aggregate_stats(args, results_keys, exp_keys)
        print(df.columns)
        print(df[['Ntraining', 'predict_testing_eval_fvaf', 'L2_regularization']])

        # Construct output file name for the created df
        fname = '%s/%s_%s_aggregate.pkl'%(args.results_path, args.label, args.exp_type)

        # Write the file
        with open(fname, "wb") as fp:
            pickle.dump(df, fp)

    else:
        # Do the work
        execute_exp(args)
