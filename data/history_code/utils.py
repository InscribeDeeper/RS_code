# -*- coding: utf-8 -*-

def load_pickle(path):
    '''
    load pickle file
    
    input: path
    
    >>> data = load_pickle('data.pk')
    '''
    from pickle import load
    with open(path, 'r') as f:
        return load(f)
    
def save_pickle(obj, path):
    '''
    save obj into pickle file
    
    input: object, path of pickle
    
    >>> save_pickle(data, 'data.pk')
    '''
    from pickle import dump
    with open(path, 'wb') as f:
        dump(obj, path)
    return None


def str2date(s):
    '''
    convert str into pandas.datetime
    
    >>> str2date('2019-1-1')
    >>> datetime.datetime(2019, 1, 1, 0, 0)
    '''
    from pandas import datetime
    date_list = s.strip().split('-')
    date_list = [int(x) for x in date_list]
    return datetime(*date_list)