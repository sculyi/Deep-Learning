import numpy as np

def strip_zeros(lst):
    if not isinstance(lst, list):
	lst=lst.tolist()
    if not 0 in lst:
	return lst
    else:
	return lst[0:lst.index(0)]
