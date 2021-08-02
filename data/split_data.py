import pdb
import os
import sys
import csv
import json
import numpy as np

def split_by_month(data):
    trn, vld, tst = [], [], []

    numdata = len(data)

    data.sort(key=lambda x:x[-1]) # sort data by date (in unix timestamp)
    
    times = np.array([float(i[-1]) for i in data])
    
    oldest_time = max(times)
    start_test_time = oldest_time - 30 * 24 * 60 * 60 # (30 days)
    start_valid_time = start_test_time - 30 * 24 * 60 * 60 # (30 days)
    
    tst_idx = times >= start_test_time
    vld_idx = (times >= start_valid_time) * (times < start_test_time)
    trn_idx = ~(tst_idx + vld_idx)

    trn_end_idx = np.where(trn_idx == True)[0].max()
    vld_end_idx = np.where(vld_idx == True)[0].max()

    trn = data[:trn_end_idx]
    vld = data[trn_end_idx:vld_end_idx]
    tst = data[vld_end_idx:]
    
    # Filter out new (cold-start) users and items
    trnusers = set([i[0] for i in trn])
    trnitems = set([i[1] for i in trn])

    vld = [row for row in vld if (row[0] in trnusers and row[1] in trnitems)]
    tst = [row for row in tst if (row[0] in trnusers and row[1] in trnitems)]

    print('\nTraining data:\t\t {}'.format(len(trn)))
    print('Validation data:\t {}'.format(len(vld)))
    print('Test data:\t\t {}'.format(len(tst)))
    print('\n# of total data:\t {} / {}'.format(len(trn) + len(vld) + len(tst), len(data)))

    return trn, vld, tst
 
fn = sys.argv[1]

print('\n'+fn+'\n')
        
data = [json.loads(l) for l in open(fn)]
# This code assumes Amazon data JSON format. The Amazon data are avaialble at 'http://jmcauley.ucsd.edu/data/amazon/'.
mydata = [[l['reviewerID'], l['asin'], l['overall'], int(float(l['unixReviewTime']))] for l in data]

trndata, vlddata, tstdata = split_by_month(mydata)

dname = fn.split('_')[1].lower()
dname = dname.split('.')[0]
    
basename = dname
if not os.path.exists(basename): os.makedirs(basename)
    
dirname = dname+'/split/'
if not os.path.exists(dirname): os.makedirs(dirname)

# Save the dataset in csv format
writer = csv.writer(open(dirname+'trn.csv', 'w'))
writer.writerows(trndata)

writer = csv.writer(open(dirname+'vld.csv', 'w'))
writer.writerows(vlddata)

writer = csv.writer(open(dirname+'tst.csv', 'w'))
writer.writerows(tstdata)

print('\nDone\n')
