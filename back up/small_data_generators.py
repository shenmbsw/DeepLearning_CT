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
        print(input.shape,ref_shape[0])
        input = input[:ref_shape[0],:,:]
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

@restartable
def dataset_2d_iterator(data_filepath, label_filepath, pad_heights=470):
    data = np.load(data_filepath)
    label = np.load(label_filepath).reshape(-1,1)
    size = label.shape[0]
    for i in range(size):
        result = np.concatenate(fast_pad(data[i:i+1],[pad_heights,512,512])).reshape((1,-1,512,512))
        yield result, label[i:i+1]

def main():
    data_filepath = 'temp.npy'
    label_filepath = 'label.npy'
    dataset_generators = {
            'train': dataset_2d_iterator(data_filepath, label_filepath),
            'test':  dataset_2d_iterator(data_filepath, label_filepath),
        }
    for epoch in range(10):
        for a, test_batch in enumerate(dataset_generators['test']):
            print(epoch,a)


if __name__=="__main__":
    main()
