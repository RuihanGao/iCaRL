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


# define NN models
class CNN(nn.Module):
    def __init__(self, seq_len):
        super(CNN, self).__init__()
        self.seq_len = seq_len
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,5))
        self.conv1_drop = nn.Dropout2d(p=0.8)

    def forward(self, x):
        #print('Conv:', x.size())
        x = self.conv1(x)
        # print('Conv', x.size())
        x = F.relu(F.max_pool2d(x, 2))
        # print('Pool', x.size())
        x = x.view(-1, 3*2*3)
        return x
    

class CNN_LSTM(nn.Module):
    def __init__(self, seq_len):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN(seq_len)
        self.lstm = nn.LSTM(
            input_size=3*2*3, 
            hidden_size=50, 
            num_layers=2,
            batch_first=True,
           dropout=0.8)
        
        self.linear = nn.Linear(50,23)
        self.hidden = []
        
        
    def init_hidden(self, h, c):
        self.hidden = (h, c)
        # Set initial hidden and cell states: initialize outside 
        #return (h, c) # (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))

    def forward(self, x):
        
        #print(x.size())
        batch_size, timesteps, C, H, W, sequence_size = x.size()
        #print(batch_size*timesteps,C, H, W, sequence_size)
        c_in = x.view(batch_size * timesteps*sequence_size, C, H, W)
        #print(c_in.size())
        
        c_out = self.cnn(c_in)
        #print(c_out.size())
        
        r_in = c_out.view(batch_size,sequence_size,-1)
        r_out, (h_n, h_c) = self.lstm( r_in, self.hidden)#(self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:] ))
        r_out2 = self.linear(r_out[:, -1, :])

        return F.log_softmax(r_out2, dim=1)

batch_size=23
num_epochs = 200
learning_rate = 0.001
log_interval = 10
save_interval  = 200
seq_len = 75

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def slide_learn(iteration):
    
    dirName = 'slide_info_' + str(iteration)
    try:
        # Create target Directory
        os.mkdir(dirName)
    except FileExistsError:
        print('Re-writing the touch')
    dirName = dirName + '/'
    
    # load data
    [train_ids, train_labels, test_ids, test_labels] = pickle.load(open('slide_6_10.pkl', 'rb'))
    print("loading data")
    print(len(train_ids), len(train_labels), len(test_ids)) # 1173 1173 253
    training_dataset = Dataset(train_ids, train_labels)
    X_train = training_dataset.get_X()
    Y_train = training_dataset.get_y()
    print("example Dataset")
    print(training_dataset[0][0].size()) # tuple e.g. (torch.Size([1, 6, 10, 75]), 0)
    # from iCaRL print(X_train.shape, Y_train.shape, X_test.shape) #(50000, 3, 32, 32) (50000,) (10000, 3, 32, 32)
    test_dataset = Dataset(test_ids, test_labels)
    X_test = test_dataset.get_X()
    Y_test = test_dataset.get_y()
    print(X_train.shape)
    print(len(X_train))
    print(Y_train[0])
    print(Y_train.shape)
    # print(X_train.shape, Y_train.shape, X_test.shape)
    train_loader = data2.DataLoader(training_dataset, batch_size=batch_size)
    test_loader = data2.DataLoader(test_dataset, batch_size=batch_size)
    
    model = CNN_LSTM(seq_len).to(device)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # set initial hidden state
    (h_ini, c_ini) = (torch.zeros(2, batch_size, 50).to(device) , torch.zeros(2, batch_size, 50).to(device))
    
    epoch_lists = []
    model.train()
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):

            data = np.expand_dims(data, axis=1)
            data = torch.FloatTensor(data)
            #print(target.size())
            data, target = data.to(device), target.to(device)


            data, target = Variable(data), Variable(target)
            #print('Data size:', data.size())

            optimizer.zero_grad()
            
            # init hidden states
            model.init_hidden(h_ini, c_ini)
            
            output = model(data)
            #print(target.size(), output.size())

            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            #if batch_idx % log_interval == 0:
        epoch_lists.append(loss.item())
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), dirName + 'model_epoch_' + str(epoch) +'.ckpt')
         
    # save epochs
    pickle.dump(epoch_lists, open(dirName + 'loss.pkl', 'wb'))
    
    ## check for accuracy
        
    results = []
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in train_loader:

        data = np.expand_dims(data, axis=1)
        data = torch.FloatTensor(data)        
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(
            output, target).item()  # sum up batch loss
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(train_loader.dataset)
    results.append( 100.0 * correct.item() / len(train_loader.dataset) )

    test_loss = 0
    correct = 0
    for data, target in test_loader:

        data = np.expand_dims(data, axis=1)
        data = torch.FloatTensor(data)        
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(
            output, target).item()  # sum up batch loss
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)

    results.append( 100.0 * correct / len(test_loader.dataset) )
    
    pickle.dump(results, open(dirName + 'results.pkl', 'wb'))

slide_learn(400)
