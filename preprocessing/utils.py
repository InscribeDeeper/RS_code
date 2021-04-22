import pandas as pd


# =============================================================================
# utils
# =============================================================================

def ts_str2sec(format_time):
    '''
    input: format_time = "2018-03-01 13:21:04"
    output: timeStamp = 1381419600
    '''
    ts = time.strptime(format_time, "%Y-%m-%d %H:%M:%S")
    return time.mktime(ts)


def ts_sec2str(timeStamp):
    '''
    input:  timeStamp = 1381419600
    output: format_time = "2018-03-01 13:21:04"
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeStamp))


def ts_attrs_add(df, ts_col='request_time'):
    # num_samples=10000
    # ######### generate request_date
    # df = read_csv(PATH_CLICK, nrows=num_samples)
    # ts_col = 'request_time'
    df[ts_col] = pd.to_datetime(df[ts_col])
    df[ts_col + '_sec'] = df[ts_col].astype(str).progress_apply(ts_str2sec)

    # For visulization
    df['hour'] = df[ts_col].dt.hour
    df['day'] = df[ts_col].dt.day
    df['month'] = df[ts_col].dt.month
    df['year'] = df[ts_col].dt.year
    df['daysinmonth'] = df[ts_col].dt.daysinmonth
    df['dayofyear'] = df[ts_col].dt.dayofyear

    # year-month-day
    df[ts_col[0:-4] + 'date'] = df[ts_col].dt.date

    return df

