import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    n = x_train.shape[0]
    alpha = torch.zeros(n,1, requires_grad=True)
    print(alpha.shape)
    print(y_train.shape)
    for _ in range(num_iters):
        loss = loss_fn(x_train, y_train, alpha, kernel, n)
        loss.requires_grad_()
        loss.backward()
        with torch.no_grad():
            alpha -= lr * alpha.grad;
            alpha.clamp_(0, c);
        alpha.grad = None
    return alpha.detach()

def loss_fn(x_train, y_train, alpha, kernel, n):
    loss = 0
    loss2 = 0
    for i in range(n):
        for j in range(n):
            K = kernel(x_train[i], x_train[j])
            loss = loss + 0.5* alpha[i] * alpha[j] * y_train[i] *y_train[j] * K
        loss2 += alpha[i]
    return loss - loss2

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    m = x_test.shape[0]
    n = x_train.shape[0]
    gram = torch.empty(n,m)
    for i in range(n):
        for j in range(m):
            gram[i][j] = kernel(x_train[i],x_test[j])
    o = alpha * y_train @ gram
    o2 = torch.Tensor(o)
    return o2
    

class CAFENet(nn.Module):
    def __init__(self):
        '''
            Initialize the CAFENet by calling the superclass' constructor
            and initializing a linear layer to use in forward().

            Arguments:
                self: This object.
        '''
        super(CAFENet, self).__init__()
        self.fc1 = nn.Linear(91200,6)
        pass
        
        
        
        

    def forward(self, x):
        '''
            Computes the network's forward pass on the input tensor.
            Does not apply a softmax or other activation functions.

            Arguments:
                self: This object.
                x: The tensor to compute the forward pass on.
        '''
        x = self.fc1(x)
        return x

def fit(net, X, y, n_epochs=5000):
    '''
    Trains the neural network with CrossEntropyLoss and an Adam optimizer on
    the training set X with training labels Y for n_epochs epochs.

    Arguments:
        net: The neural network to train
        X: n x d tensor
        y: n x 1 tensor
        n_epochs: The number of epochs to train with batch gradient descent.

    Returns:
        List of losses at every epoch, including before training
        (for use in plot_cafe_loss).
    
    '''
    losslist = []
    loss = nn.CrossEntropyLoss()
    param = net.parameters()
    optimizer = optim.Adam(param)
    for epochs in range(n_epochs):
        Y = net(X)
        risk = loss(Y,y)
        risk.backward()
        optimizer.step()
        optimizer.zero_grad()
        losslist.append(risk.item())
    return losslist





class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        1) a 2D convolutional layer with 1 input channel and 8 outputs, with a kernel size of 3, followed by
        2) a 2D maximimum pooling layer, with kernel size 2
        3) a 2D convolutional layer with 8 input channels and 4 output channels, with a kernel size of 3
        4) a fully connected (Linear) layer with 4 inputs and 10 outputs
        '''
        super(DigitsConvNet, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3)
        self.fc2 = nn.MaxPool2d(kernel_size=2)
        self.fc3 = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=3)
        self.fc4 = nn.Linear(4,10)
        pass

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        xb = torch.reshape(xb, (-1, 1, 8, 8))
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        xb = F.relu(self.fc3(xb))
        xb = torch.reshape(xb,(-1,4))
        xb = self.fc4(xb)
        return xb

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []

    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss
    
    
    for epoch in range(n_epochs):
        for (i, (xb,yb)) in enumerate(train_dl):
            hw2_utils.train_batch(net, loss_func, xb, yb, opt=optimizer)
        with torch.no_grad():
            loss1 = hw2_utils.epoch_loss(net, loss_func, train_dl)
            loss2 = hw2_utils.epoch_loss(net, loss_func, test_dl)
            loss1.requires_grad_()
            loss2.requires_grad_()
            loss1.backward()
            loss2.backward()
            train_losses.append(loss1.item())
            test_losses.append(loss2.item())
        
    
    
    
        

    return train_losses, test_losses

def fit_and_evaluate_decay(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []

    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss
    
    schedule = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    for epoch in range(n_epochs):
        schedule.step()
        for (i, (xb,yb)) in enumerate(train_dl):
            hw2_utils.train_batch(net, loss_func, xb, yb, opt=optimizer)
        with torch.no_grad():
            loss1 = hw2_utils.epoch_loss(net, loss_func, train_dl)
            loss2 = hw2_utils.epoch_loss(net, loss_func, test_dl)
            loss1.requires_grad_()
            loss2.requires_grad_()
            loss1.backward()
            loss2.backward()
            train_losses.append(loss1.item())
            test_losses.append(loss2.item())
            
        
        
    
    
    
        

    return train_losses, test_losses
