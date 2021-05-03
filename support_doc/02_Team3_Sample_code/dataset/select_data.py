# -*- coding: utf-8 -*-
import sys
sys.path.append('../')



def select_timerange(df, time1, time2, attr='request_time'):
    '''
    select data within a time range
    
    >>> select_timerange(DataFrame, '2019-1-1', '2019-2-1')
    '''
    from utils import str2date
    time1 = str2date(time1)
    time2 = str2date(time2)
    expr = '({0}>=@time1) and ({0}<=@time2)'.format(attr)
    return df.query(expr)









