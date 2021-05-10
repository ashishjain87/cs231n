import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    
    # LABELS
    parser.add_argument('--new-labels', action='store_true')        # Code assumes you already have a pickled file, generate new if needed
    parser.add_argument('--labels-pickle')                          # Don't need to write this flag

    # DATA SPLIT
    # Generate new data split if needed.
    # If you don't use this flga, the code assumes you have already generated 3 csv ids file of form
    #       "train_split.csv"
    #       "val_split.csv"
    #       "test_split.csv"
    parser.add_argument('--new-data-split', action='store_true') 

    # DATA: currently unsupported
    # parser.add_argument('--data-pickle', action='store_true')
    
    # LOCAL MACHINE
    parser.add_argument('--local', action='store_true')

    args = parser.parse_args()
    return args