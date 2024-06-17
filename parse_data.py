import csv
import numpy as np
import random
import chardet





def filter_data():
    rows_to_consider = []

   # b = pd.read_csv(csv, encoding=encoding, header=None, sep=',', engine='python')
    with open('data_climate.csv', newline='', encoding='utf-8') as csv_file:
        r = csv.reader(csv_file, delimiter=';')
        discard = r.__next__()

        for i in r:
            #print(i)

            if i[27] != "weiß nicht, keine Angabe":
                rows_to_consider.append(i)

    return rows_to_consider

def get_y(data):
    y = np.zeros(len(data))
    ind = 0
    switch = {
        'oder gehen sie nicht weit genug' : 1,
        'angemessen sind' : 0,
        'oder zu weit gehen' : -1
    }
    for i in data:
        y[ind] = switch.get(i[27])
        ind+=1
    return y

def get_x(data):
    x = np.zeros((4, len(data)))
    x[0] = np.ones(len(data))
    switch_price = {
        "zu hoch": 1,
        "zu niedrig": -1,
        "angemessen": 0,
        "weiß nicht, keine Angabe": 0
    }
    switch_worries ={
        "sehr große" :3,
        "große": 2,
        "weniger große": 1,
        "oder gar keine Sorgen": 0,
        "weiß nicht, keine Angabe": 0
    }
    switch_agency = {
        "sehr gut": 2,
        "eher gut": 1,
        "weniger gut": -1,
        "gar nicht gut": -1,
        "weiß nicht, keine Angabe": 0
    }
    ind = 0
    for i in data:
        x[1][ind] = switch_price.get(i[21]) + switch_price.get(i[22]) + switch_price.get(i[24])
        x[2][ind] = switch_worries.get(i[26])
        x[3][ind] = switch_agency.get(i[28])
        ind += 1

    return x

def get_x_2(data, n):
    x = np.zeros((12, n))
    x_test= np.zeros((12, len(data)-n))
    x[0] = np.ones(n)
    x_test[0] = np.ones(len(data)-n)
    switch_worries = {
        "sehr große": 5,
        "große": 6,
        "weniger große": 7,
        "oder gar keine Sorgen": 8,
        "weiß nicht, keine Angabe": 0
    }
    switch_agency = {
        "sehr gut": 1,
        "eher gut": 2,
        "weniger gut": 3,
        "gar nicht gut": 4,
        "weiß nicht, keine Angabe": 0
    }
    switch_price = {
        "zu hoch": 9,
        "zu niedrig": 11,
        "angemessen": 10,
        "weiß nicht, keine Angabe": 0
    }
    ind= 0
    for i in data:
        p1 = switch_price.get(i[21])
        p2 = switch_price.get(i[22])
        p3 = switch_price.get(i[24])
        if ind<n:
            if p1==p2: #at least two have to be the same
                x[p1][ind] = 1
            elif p1==p3:
                x[p1][ind] = 1
            elif p2==p3:
                x[p2][ind] = 1
            x[switch_agency.get(i[28])][ind] = 1
            x[switch_worries.get(i[26])][ind] = 1
        else:
            if p1==p2: #at least two have to be the same
                x_test[p1][ind-n] = 1
            elif p1==p3:
                x_test[p1][ind-n] = 1
            elif p2==p3:
                x_test[p2][ind-n] = 1
            x_test[switch_agency.get(i[28])][ind-n] = 1
            x_test[switch_worries.get(i[26])][ind-n] = 1

        ind += 1


    return x, x_test

def get_y_2(data, n):
    y = np.zeros(n)
    y_test = np.zeros(len(data)-n)
    ind = 0
    switch = {
        'oder gehen sie nicht weit genug' : 1,
        'angemessen sind' : 0,
        'oder zu weit gehen' : -1
    }

    for i in data:
        if ind < n:
            y[ind] = switch.get(i[27])
            ind += 1
        else:
            y_test[ind-n] = switch.get(i[27])
            ind += 1
    return y, y_test


def get_y_2_only_less(data, n):
    y = np.zeros(n)
    y_test = np.zeros(len(data)-n)
    ind = 0
    switch = {
        'oder gehen sie nicht weit genug' : 0,
        'angemessen sind' : 0,
        'oder zu weit gehen' : 1
    }

    for i in data:
        if ind < n:
            y[ind] = switch.get(i[27])
            ind += 1
        else:
            y_test[ind-n] = switch.get(i[27])
            ind += 1
    return y, y_test







#get data where the answers are quantified
def get_data_quant():
    d = filter_data()
    y = get_y(d)
    x = get_x(d)
    return x, y

#get data where the answers are used as categorical variables
def get_data_cat():
    d = filter_data()
    y, ytest = get_y_2(d)
    x, xtest = get_x_2(d)
    return x, y

def get_data_cat_test(n_train):
    n = n_train
    d = filter_data()
    random.shuffle(d)
    y, y_test = get_y_2(d, n)
    x, x_test = get_x_2(d, n)
    return x, x_test, y, y_test

def get_data_cat_test_only_less(n_train):
    n = n_train
    d = filter_data()
    random.shuffle(d)
    y, y_test = get_y_2_only_less(d, n)
    x, x_test = get_x_2(d, n)
    return x, x_test, y, y_test





