import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Class for the user-defined neural network in pytorch for training
class CustomModel(nn.Module):
    def __init__(self, hidden_layers_dim, dropout_rate):
        super(CustomModel, self).__init__()
        
        self.num_hidden = len(hidden_layers_dim)
        
        # define layer to flatten the image
        self.flatten = nn.Flatten()
        
        # define the hidden layers with user-defined size
        self.hidden_layers = nn.ModuleList()
        for inx, dim in enumerate(hidden_layers_dim):
            if inx == 0:
                self.hidden_layers.append(nn.Linear(28*28, hidden_layers_dim[inx]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_layers_dim[inx - 1], hidden_layers_dim[inx]))
                
        # define the dropout rate for hidden layers
        self.dropout_rate = nn.ModuleList()
        if dropout_rate != "None":
            if len(dropout_rate) == 1:
                for i in range(self.num_hidden):
                    self.dropout_rate.append(nn.Dropout(dropout_rate[0]))
            else:
                for rate in dropout_rate:
                    if rate == 1:
                        self.dropout_rate.append(nn.Dropout(0))
                    else:
                        self.dropout_rate.append(nn.Dropout(rate))
        else:
            for i in range(len(hidden_layers_dim)):
                self.dropout_rate.append(nn.Dropout(0))
        
        # Define output layer
        self.output_layer = nn.Linear(hidden_layers_dim[-1], 10)
        
        # Define activation function (e.g., ReLU)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # define how we will forward a new vector into the nn
        x = self.flatten(x)
        for i in range(self.num_hidden):
            x = self.dropout_rate[i](x)
            x = self.hidden_layers[i](x)
            x = self.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return(x)
    
# define the custom training function that also allow the update of the current progress to the flask app
def train(model, optimizer, current_epochs, num_epochs, total_num_epochs, starting_time, device, train_loader, running_experiments, socketio = None):
    # prepare the criterion
    criterion = nn.CrossEntropyLoss()
    # training the model
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # current progress, not update if we do not update to socketio
        if socketio is not None: 
            progress = 100 * current_epochs / total_num_epochs
            running_experiments[starting_time]["progress"] = progress
            current_epochs += 1
            socketio.emit('progress_update', {'starting_time': starting_time, 'progress': str(progress) + "%"}, namespace='/update')
    return model, current_epochs

# custom function to test the model
def test(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total