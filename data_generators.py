import numpy as np
import copy
import os
import math
try:
    from itertools import izip as zip
except ImportError:
    print('This is Python 3')

def fast_pad(input, ref_shape):
    assert len(ref_shape)==3
    if ref_shape[0]>=input[0].shape[0]:
        for i in range(len(input)):
            input[i] = np.lib.pad(input[i],((0,ref_shape[0]-input[i].shape[0]),(0,ref_shape[1]-input[i].shape[1]),(0,ref_shape[2]-input[i].shape[2])),'constant',constant_values = 0)
    else:
        for i in range(len(input)):
            input[i] = np.lib.pad(input[i],((0,0),(0,ref_shape[1]-input[i].shape[1]),(0,ref_shape[2]-input[i].shape[2])),'constant',constant_values = 0)
        input[0] = input[0][:ref_shape[0],:,:]
    return input

def concatenate_list(array_list):
    return np.stack(array_list)

class GeneratorRestartHandler(object):
    def __init__(self, gen_func, argv, kwargv):
        self.gen_func = gen_func
        self.argv = copy.copy(argv)
        self.kwargv = copy.copy(kwargv)
        self.local_copy = self.gen_func(*self.argv, **self.kwargv)

    def __iter__(self):
        return GeneratorRestartHandler(self.gen_func, self.argv, self.kwargv)

    def __next__(self):
        return next(self.local_copy)

    def next(self):
        return self.__next__()

def restartable(g_func):
    def tmp(*argv, **kwargv):
        return GeneratorRestartHandler(g_func, argv, kwargv)

    return tmp


# All the data are saved in 2 file called processed_train_data and processed_tese_data
# The data set is saved as data_filepath+str(i)+'.npy'
def creat_datapath(data_filepath, label_filepath, num):
    data_pathlist = []
    label_pathlist = []
    for i in range(1, num+1):
        data_pathlist.append(data_filepath+str(i)+'.npy')
        label_pathlist.append(label_filepath+str(i)+'.npy')
    return data_pathlist, label_pathlist


@restartable
def dataset_2d_iterator(data_pathlist, label_pathlist, pad_heights=470):
    for i in range(len(data_pathlist)):
        data = np.load(data_pathlist[i])
        label = np.load(label_pathlist[i]).reshape(-1,1)
        size = label.shape[0]
        for i in range(size):
            result = np.concatenate(fast_pad(data[i:i+1],[pad_heights,512,512])).reshape((1,-1,512,512))
            yield result, label[i:i+1]

def main():
    data_filepath = 'processed_train_data/train_data'
    labels_filepath = 'processed_train_data/train_labels'

    train_data_pathlist, train_labels_pathlist = dg.creat_datapath(data_filepath, labels_filepath, 1)
    dataset_generators = dataset_2d_iterator(train_data_pathlist, train_labels_pathlist)

    for epoch in range(10):
        for a, test_batch in enumerate(dataset_generators['test']):
            print(epoch,a)


if __name__=="__main__":
    main()
