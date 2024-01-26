from flask import Flask, render_template, request, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from threading import Event
from threads import CustomThread
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from models import CustomModel, train, test
from test_numbers import test_int, test_float
from experiments import ExperimentResult
from bisect import insort_left
from datetime import datetime
import os
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.secret_key = 'your_secret_key'
socketio = SocketIO(app)

# Initial parameters
initial_params = {
    "hidden_layers_dim": "128",
    "dropout_rate": "None",
    "optimizer": "Adam",
    'learning_rate': "0.001",
    'batch_size': "64",
    'num_epochs': "10",
    'num_folds': "None"
}

# store different choice of optimizer
optimizer_dict = {
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
    "Rprop": optim.Rprop,
    "SGD": optim.SGD 
}

# List to store experiment results
finished_experiments = []

# (Current time - params) to store currently running jobs
running_experiments = {}

# We also need dictionary to store current threads as well as the event associated to them in order to stop a thread gracefully when user cancel a thread
running_experiments_stop_event = {} # (Current time - stop event)

# our dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))]) # transform and normalize
train_dataset = MNIST(os.getcwd(), download=True, transform=transform, train=True)
test_dataset = MNIST(os.getcwd(), download=True, transform=transform, train=False)

def run_experiment_async(params, starting_time):
    # extract different choice of parameters
    lr_lst = params["learning_rate"]
    bs_lst = params["batch_size"]
    ne_lst = params["num_epochs"]
    
    # extract number of folds for cv
    num_folds = params["num_folds"]
    
    # store resulting parameters
    best_lr = best_bs = best_ne = best_acc = best_cv_acc = float("-inf")
    
    
    # consider the case of not using cv, only when we have 1 combination of parameters
    if num_folds == "None":
        # train the data
        # Create a PyTorch model and run the experiment
        model = CustomModel(params["hidden_layers_dim"], params["dropout_rate"]) 
        # # prepare the setting for training and testing
        train_loader = DataLoader(dataset = train_dataset, batch_size = bs_lst[0], shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = bs_lst[0], shuffle = True)
        # initialize optimizer
        optimizer = optimizer_dict[params["optimizer"]](model.parameters(), lr = lr_lst[0])
        # train the model 
        model, current_epochs = train(model, optimizer, 1, ne_lst[0], ne_lst[0], starting_time, device, train_loader, running_experiments, socketio)
        # We enter the testing part
        running_experiments[starting_time]["progress"] = "Final testing..."
        socketio.emit('progress_update', {'starting_time': starting_time, 'progress': "Final testing..."}, namespace='/update')
        # Now we will test the model on the evaluation set
        best_acc = test(model, test_loader, device)
        best_cv_acc = "None"
        best_lr = lr_lst[0]
        best_bs = bs_lst[0]
        best_ne = ne_lst[0]
    else: 
        # get total number of epochs: sum of different choice of epochs * number of comb of lr and batch size
        total_num_epochs = sum(ne_lst) * len(lr_lst) * len(bs_lst) * num_folds
        current_epochs = 1 # get the current epochs for later update of current progress
        # split the training data into kfold
        training_kfold = KFold(n_splits = num_folds, shuffle = True, random_state = 42)
        # perform grid search on these lists with kfolds
        for batch_size in bs_lst:
            for learning_rate in lr_lst:
                for num_epochs in ne_lst:
                    curr_cv_acc = 0 # current total accuracy of all folds for calculation of mean accuracy across all folds
                    for fold, (train_ids, test_ids) in enumerate(training_kfold.split(train_dataset)):
                        # Create a PyTorch model and run the experiment
                        model = CustomModel(params["hidden_layers_dim"]) 
                        # Sample elements randomly from a given list of ids, no replacement.
                        train_subsampler = SubsetRandomSampler(train_ids)
                        test_subsampler = SubsetRandomSampler(test_ids)
                        # Define data loaders for training and testing data in this fold
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler)
                        test_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_subsampler)
                        # initialize optimizer
                        optimizer = optimizer_dict[params["optimizer"]](model.parameters(), lr = learning_rate)
                        # train the model and update current_epochs for later update of progress
                        model, current_epochs = train(model, optimizer, current_epochs, num_epochs, total_num_epochs, starting_time, device, train_loader, running_experiments, socketio)
                        # Now we will test the model on the evaluation set
                        accuracy = test(model, test_loader, device)
                        # add to curr_cv_acc for calculation of final score later
                        curr_cv_acc += accuracy
                    # check if current accuracy is the best accuracy
                    curr_cv_acc /= num_folds
                    if curr_cv_acc > best_cv_acc:
                        best_cv_acc = curr_cv_acc
                        best_lr = learning_rate
                        best_bs = batch_size
                        best_ne = num_epochs
        # We enter the testing part
        running_experiments[starting_time]["progress"] = "Final testing..."
        socketio.emit('progress_update', {'starting_time': starting_time, 'progress': "Final testing..."}, namespace='/update')
        # Now we evaluate on the actual test data using both full train and test
        # Create a PyTorch model and run the experiment
        model = CustomModel(params["hidden_layers_dim"]) 
        # make train and test loader
        train_loader = DataLoader(train_dataset, batch_size=best_bs, shuffle = True)
        test_loader = DataLoader(test_dataset, batch_size=best_bs, shuffle = True)
        # initialize optimizer
        optimizer = optimizer_dict[params["optimizer"]](model.parameters(), lr = best_lr)
        # We train the model with the best parameters on the training data
        model, current_epochs = train(model, optimizer, 1, best_ne, best_ne, starting_time, device, train_loader, running_experiments)
        # Now we will test the model on the evaluation set
        best_acc = test(model, test_loader, device)

    # We need to store the best combination of parameters to show in the report
    best_params = {
        "learning_rate": best_lr,
        "batch_size": best_bs,
        "num_epochs": best_ne,
        "optimizer": params["optimizer"]
    }
    
    # Save the experiment results
    experiment_result = ExperimentResult(params, best_params, best_cv_acc, -best_acc) # since python use min heap
    
    # add the job to the finished queue and remove from running queue
    del running_experiments[starting_time]
    insort_left(finished_experiments, experiment_result)
    socketio.emit('progress_complete', {'starting_time': starting_time, 'params': params, "best_params": best_params, "cv_acc": best_cv_acc, "accuracy": best_acc}, namespace='/update')
    
@app.route('/cancel_experiment/<experiment_id>', methods=['POST'])
def cancel_experiment(experiment_id):
    # Function to cancel the experiment with the given ID (which is also the starting time)
    if experiment_id in running_experiments:
        # Stop the thread by get the event and active the event
        experiment_stop_event = running_experiments_stop_event[experiment_id]
        experiment_stop_event.set()
        del running_experiments_stop_event[experiment_id]
        # we delete the experiment from the dictionary
        del running_experiments[experiment_id]

    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html', 
                           running_experiments = running_experiments, 
                           finished_experiments = finished_experiments, 
                           initial_params = initial_params)

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    # We first need to test the input format
    error_messages = [] # store all error messages for this test
    
    # test the hidden layers
    hidden_layers_dim_input = request.form["hidden_layers_dim"].replace(" ", "").split(",")
    if not all(test_int(h) for h in hidden_layers_dim_input):
        error_messages.append('List of dimensions of hidden layers must only contains integers')
    
    # test the dropout rate
    dropout_rate_input = request.form["dropout_rate"].replace(" ", "")
    if dropout_rate_input != "None":
        dropout_rate_input = dropout_rate_input.split(",")
        if not all(test_float(d) for d in dropout_rate_input):
            error_messages.append("List of dropout rates must only contain floats")
        if len(dropout_rate_input) != 1 and len(dropout_rate_input) != len(hidden_layers_dim_input):
            error_messages.append("List of dropout rates must only have 1 number of have same number of numbers as number of hidden layers")
    
    # test the optimizer
    optimizer_input = request.form["optimizer"].replace(" ", "")
    if optimizer_input not in optimizer_dict:
        error_messages.append(f"Optimizer should be one of: {optimizer_dict.values()}")
    
    # test the learning rate
    learning_rate_input = request.form["learning_rate"].replace(" ", "").split(",")
    if not all(test_float(l) for l in learning_rate_input):
        error_messages.append('List of learning rates must only contains floats')
    
    # test the batch size
    batch_size_input = request.form["batch_size"].replace(" ", "").split(",")
    if not all(test_int(b) for b in batch_size_input):
        error_messages.append('List of batch sizes must only contains integers')
    
    # test the number of epochs
    num_epochs_input = request.form["num_epochs"].replace(" ", "").split(",")
    if not all(test_int(b) for b in  num_epochs_input):
        error_messages.append('List of number of epochs must only contains integers')
    
    # test the number of fold
    num_folds_input = request.form["num_folds"]
    if not (test_int(num_folds_input) or num_folds_input == "None"):
        error_messages.append('the number of folds must be a single integers or None')
        
    # check if we have any error messages
    if len(error_messages) > 0:
        for e in error_messages:
            flash(e)
        return redirect(url_for('index'))
    
    # we consider each parameter, except the layers of the nn, to be a list of different choices for grid search
    params = {
        "hidden_layers_dim": list(map(int, hidden_layers_dim_input)),
        "dropout_rate": dropout_rate_input if dropout_rate_input == "None" else list(map(float, dropout_rate_input)),
        "optimizer": optimizer_input,
        'learning_rate': list(map(float, learning_rate_input)),
        'batch_size': list(map(int, batch_size_input)),
        'num_epochs': list(map(int, num_epochs_input)),
        'num_folds': int(num_folds_input) if test_int(num_folds_input) else "None",
    }

    # We only consider the case of num_folds = -1 if we have only 1 combination of parameters
    if len(params["learning_rate"]) * len(params["batch_size"]) * len(params["num_epochs"]) > 1 and params["num_folds"] == "None":
        flash('Cannot perform grid search with no cross validation')
        return redirect(url_for('index'))
    
    # Check if exactly the same job has been run
    if (params in [f.params for f in finished_experiments]) or (params in [running_experiments[r]["params"] for r in running_experiments]):
        return redirect(url_for('index'))

    # Add the job to the list of running jobs
    starting_time  = str(datetime.now()) # this will also be the id of the experiment
    running_experiments[starting_time] = {
        "params": params,
        "progress": 0
    }
    
    # get initial progress
    socketio.emit('progress_update', {'starting_time': starting_time, 'progress': 0}, namespace='/update')
    
    # new thread for running the task, with event for stopping the thread
    # currently the thread are running concurrently 
    experiment_stop_event = Event()
    running_experiments_stop_event[starting_time] = experiment_stop_event
    experiment_thread = CustomThread(target=run_experiment_async, args=(params, starting_time), stop_event = experiment_stop_event)
    experiment_thread.start()

    return redirect(url_for('index'))

@socketio.on('connect', namespace='/update')
def handle_connect():
    print('Client connected')

if __name__ == '__main__':
    socketio.run(app, debug=True)