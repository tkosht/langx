import datetime


def now():
    return datetime.datetime.now().strftime("%Y/%m/%d %T")


def log_info(*args, **kwargs):
    print(now(), "INFO", *args, **kwargs)
