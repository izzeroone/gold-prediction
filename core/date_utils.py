import datetime as dt


def next_date(cur_date):
    n_date = dt.datetime.strptime(cur_date, '%Y-%m-%d')
    n_date = n_date + dt.timedelta(days=1)
    while n_date.date().weekday() == 5 or n_date.date().weekday() == 6: #Rest day
        n_date = n_date + dt.timedelta(days=1)
    return n_date.date()