import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import copy

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model

#############################################################################
# Process                                                                   #
#############################################################################
# - There is a fixed set of K clients, each with a fixed local dataset.
# [Start round]
# 1. A random fraction of C of clients is selected, and the server sends the current global model.
# 2. Each selected client then performs local computation based on the global state and its dataset.
# 3. Then each client sends an update to the server. 

#############################################################################
# Hyperparameters                                                           #
#############################################################################
NUM_CLIENTS     = 60
FRACTION        = 0.1
L_BATCH_SIZE    = 10
L_EPOCH         = 10
LR              = 1e-2

DIR_DATA        = './mnist_data/'

# CUDA
CUDA = False
CUDA_INDEX = 2

#############################################################################
# Util functions                                                            #
#############################################################################

def cuda_device(enable=True, idx=0):
    if enable:
        d_str = 'cuda:' + str(idx)
        device = torch.device(d_str if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    return device


def get_train_dataset(split=True):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_torch = datasets.MNIST(
        DIR_DATA, train=True, download=True, transform=tf
    )

    loader = DataLoader(dataset_torch, batch_size=len(dataset_torch), shuffle=True)

    images, labels = next(enumerate(loader))[1]
    images, labels = images.numpy(), labels.numpy()

    images = np.split(images, NUM_CLIENTS)
    labels = np.split(labels, NUM_CLIENTS)

    return images, labels


def get_test_dataset():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_torch = datasets.MNIST(
        DIR_DATA, train=False, download=True, transform=tf
    )

    loader = DataLoader(dataset_torch, batch_size=len(dataset_torch), shuffle=True)

    images, labels = next(enumerate(loader))[1]
    images, lables = images.numpy(), labels.numpy()

    return images, labels


#############################################################################
# Central                                                                   #
#############################################################################
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def test_model(model):
    device = torch.device(next(model.parameters()).device if next(model.parameters()).is_cuda else 'cpu')

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    images, labels = get_test_dataset()
    images, labels = torch.tensor(images), torch.tensor(labels)
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    batch_loss = nn.NLLLoss()(outputs, labels)
    loss += batch_loss.item()

    _, pred_labels = torch.max(outputs, 1)
    pred_labels = pred_labels.view(-1)
    correct += torch.sum(torch.eq(pred_labels, labels)).item()
    total += len(labels)

    # for batch_idx, (x, y) in enumerate(testloader):
    #     x, y = x.to(device), y.to(device)

    #     # Inference
    #     outputs = model(x)
    #     batch_loss = nn.NLLLoss()(outputs, y)
    #     loss += batch_loss.item()

    #     # Prediction
    #     _, pred_labels = torch.max(outputs, 1)
    #     pred_labels = pred_labels.view(-1)
    #     correct += torch.sum(torch.eq(pred_labels, y)).item()
    #     total += len(y)

    accuracy = correct/total
    return accuracy, loss


def central(dataset_queues, model_queues, update_queues):
    device = cuda_device(CUDA, CUDA_INDEX)

    # Create model
    model = Model()
    model.to(device)

    #############################################################################
    # Select random fraction of clients, and global model to each client        #
    #############################################################################
    train_images, train_labels = get_train_dataset(split=True)
    for i in range(NUM_CLIENTS):
        dataset_queues[i].put((train_images[i].copy(), train_labels[i].copy()))

    num_rounds = 0
    while True:
        #############################################################################
        # Test                                                                      #
        #############################################################################
        print(test_model(model))
        
        print('# of Round : ', num_rounds)

        #############################################################################
        # Select random fraction of clients, and global model to each client        #
        #############################################################################
        model.train()

        num_cli_sel = max(int(FRACTION * NUM_CLIENTS), 1)
        clients_sel = np.random.choice(range(NUM_CLIENTS), num_cli_sel, replace=False)
        print('selected clients : ', clients_sel)

        for idx in clients_sel:
            model_state = model.state_dict()
            model_queues[idx].put((model_state))

        #############################################################################
        # Receieve updates from clients                                             #
        #############################################################################
        local_states = []
        local_losses = []
        for idx in clients_sel:
            lstate, loss = update_queues[idx].get()
            local_states.append(lstate)
            local_losses.append(loss)

        #############################################################################
        # Train                                                                     #
        #############################################################################
        averaged_states = average_weights(local_states)
        model.load_state_dict(averaged_states)

        num_rounds += 1
        print()

#############################################################################
# Local                                                                     #
#############################################################################
def train_step(model, images, labels):
    device = model.getdevice()

    images = torch.tensor(images.copy(), device=device)
    labels = torch.tensor(labels.copy(), device=device)

    optimizer = optim.Adam(model.parameters(), LR)
    optimizer.zero_grad()
    probs = model(images)
    logprobs = torch.log(probs)
    loss = nn.NLLLoss()(logprobs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def local(i, dataset_queue, model_queue, update_queue):
    device = cuda_device(CUDA, CUDA_INDEX)

    model = Model()
    model.to(device)

    train_images, train_labels = dataset_queue.get()

    while True:
        #############################################################################
        # Receieve global model                                                     #
        #############################################################################
        global_state = model_queue.get()
        model.load_state_dict(global_state)

        #############################################################################
        # Local Computation (train)                                                 #
        #############################################################################
        losses = []
        for epoch in range(L_EPOCH):
            loss = train_step(model, train_images, train_labels)
            losses.append(loss)

        #############################################################################
        # Send model to the server                                                  #
        #############################################################################
        local_state = copy.deepcopy(model.state_dict())
        update_queue.put((local_state, losses))


if __name__  == '__main__':
    ############################################################################
    # Create queues for communication                                          #
    ############################################################################
    dataset_queues = []
    model_queues = []
    update_queues = []
    for i in range(NUM_CLIENTS):
        dataset_queues.append(mp.Queue(1))
        model_queues.append(mp.Queue(1))
        update_queues.append(mp.Queue(1))

    ############################################################################
    # Create processes                                                         #
    ############################################################################
    coordinator = mp.Process(target=central, args=(dataset_queues, model_queues, update_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_CLIENTS):
        agents.append(mp.Process(target=local, args=(i, dataset_queues[i], model_queues[i], update_queues[i])))

    for i in range(NUM_CLIENTS):
        agents[i].start()