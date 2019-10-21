import numpy as np 
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm
from lasagne.layers import Layer
from scipy.spatial.distance import cdist


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.utils import data as data2
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import os

###################### Load the data #######################

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

### Load slide dta  ###
class Dataset(data2.Dataset):
    def __init__(self, list_IDs, labels):
        # initialize
        self.labels = labels
        self.list_IDs = list_IDs
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('slide_6_10/' + ID + '.pt')
        X.unsqueeze_(0)
        y = self.labels[ID]

        return X, y
    
    def get_X(self):
        X = []
        for i in range(len(self.list_IDs)):
            ID = self.list_IDs[i]
            # x = torch.load('slide_6_10/' + ID + '.pt').unsqueeze_(0)
            x = torch.load('slide_6_10/' + ID + '.pt')
            # convert to np array
            X.append(x.numpy())
        return np.array(X)
    
    def get_y(self):
        y = []
        for i in range(len(self.list_IDs)):
            ID = self.list_IDs[i]
            y.append(self.labels[ID])
        return np.array(y)


def load_slide_data(val_ratio):
    # load data
    [train_ids, train_labels, test_ids, test_labels] = pickle.load(open('slide_6_10.pkl', 'rb'))
    print("loading slide data")
    # print(len(train_ids), len(train_labels), len(test_ids)) # 1173 1173 253
    training_dataset = Dataset(train_ids, train_labels)
    X = training_dataset.get_X()
    Y = training_dataset.get_y()
    train_num = len(X)*(1-val_ratio)
    X_train = X[:int(train_num), :, :, :]
    Y_train = Y[:int(train_num)]
    X_valid = X[int(train_num):, :, :, :]
    Y_valid = Y[int(train_num):]

    # print("example Dataset")
    # print(training_dataset[0][0].size()) # tuple e.g. (torch.Size([1, 6, 10, 75]), 0)
    # from iCaRL print(X_train.shape, Y_train.shape, X_test.shape) #(50000, 3, 32, 32) (50000,) (10000, 3, 32, 32)
    test_dataset = Dataset(test_ids, test_labels)
    X_test = test_dataset.get_X()
    Y_test = test_dataset.get_y()
    # print(X_train.shape) # (1173, 6, 10, 75)
    # print(Y_train.shape) # (1173,)

    # X_test  = x[50000:,:,:,:]
    # Y_test  = y[50000:]
    # print("dataset size")
    print(X_train.shape, Y_train.shape, X_test.shape) 
    # cifar: (50000, 3, 32, 32) (50000,) (10000, 3, 32, 32)
    # slide: (938, 6, 10, 75) (938,) (253, 6, 10, 75)
    return dict(
        X_train = lasagne.utils.floatX(X_train),
        Y_train = Y_train.astype('int32'),
        X_valid = lasagne.utils.floatX(X_valid),
        Y_valid = Y_valid.astype('int32'),
        X_test  = lasagne.utils.floatX(X_test),
        Y_test  = Y_test.astype('int32'),)


###################### Build the neural network model #######################

def build_cnn(input_var=None, n=5):
    # This block of code for the architecture and the data augmentation is inspired from Lasagne recipe code : https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False,last=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters
        
        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                if last:
                    block = ElemwiseSumLayer([stack_2, projection])
                else:
                    block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                if last:
                    block = ElemwiseSumLayer([stack_2, padding])
                else:
                    block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            if last:
                block = ElemwiseSumLayer([stack_2, l])
            else:
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
        
        return block
    
    # Building the network
    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    
    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l)
    
    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)
    
    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n-1):
        l = residual_block(l) 
    
    l = residual_block(l,last=True)
    # average pooling
    l = GlobalPoolLayer(l)
    # fully connected layer
    network = DenseLayer(
            l, num_units=100,
            W=lasagne.init.HeNormal(),
            nonlinearity=lasagne.nonlinearities.sigmoid)
     
    return network,l

############################## Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper : 
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                # Cropping and possible flipping
                if (np.random.randint(2) > 0):
                    random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
                else:
                    random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)][:,:,::-1]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]
        
        yield inp_exc, targets[excerpt]


def accuracy_measure(X_valid, Y_valid, class_means, val_fn, top1_acc_list, iteration, iteration_total, type_data):
    
    stat_hb1   = []
    stat_icarl = []
    stat_ncm   = []
    
    for batch in iterate_minibatches(X_valid, Y_valid, min(500,len(X_valid)), shuffle=False):
        inputs, targets_prep = batch
        targets = np.zeros((inputs.shape[0],100),np.float32)
        targets[range(len(targets_prep)),targets_prep.astype('int32')] = 1.
        err,pred,pred_inter = val_fn(inputs, targets)
        pred_inter  = (pred_inter.T/np.linalg.norm(pred_inter.T,axis=0)).T
        
        # Compute score for iCaRL
        sqd         = cdist(class_means[:,:,0].T, pred_inter, 'sqeuclidean')                    
        score_icarl = (-sqd).T
        # Compute score for NCM
        sqd         = cdist(class_means[:,:,1].T, pred_inter, 'sqeuclidean')                    
        score_ncm   = (-sqd).T
        
        # Compute the accuracy over the batch
        stat_hb1   += ([ll in best for ll, best in zip(targets_prep.astype('int32'), np.argsort(pred, axis=1)[:, -1:])])
        stat_icarl += ([ll in best for ll, best in zip(targets_prep.astype('int32'), np.argsort(score_icarl, axis=1)[:, -1:])])
        stat_ncm   += ([ll in best for ll, best in zip(targets_prep.astype('int32'), np.argsort(score_ncm, axis=1)[:, -1:])])
    
    print("Final results on "+type_data+" classes:")
    print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(np.average(stat_icarl)* 100))
    print("  top 1 accuracy Hybrid 1       :\t\t{:.2f} %".format(np.average(stat_hb1)* 100))
    print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(np.average(stat_ncm)* 100))
    
    top1_acc_list[iteration,0,iteration_total] = np.average(stat_icarl) * 100
    top1_acc_list[iteration,1,iteration_total] = np.average(stat_hb1) * 100
    top1_acc_list[iteration,2,iteration_total] = np.average(stat_ncm) * 100
    
    return top1_acc_list
