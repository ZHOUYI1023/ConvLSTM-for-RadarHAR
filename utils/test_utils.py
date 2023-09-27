import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .eval import calculate_accuracy



def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0

    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
  
        
    epoch_loss /= len(iterator)
    epoch_acc /= len(iterator)

        
    return epoch_loss, epoch_acc


def evaluate_uncertainty(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0

    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            
            loss = criterion(y_pred, y.repeat(model.num_estimators))

            acc = calculate_accuracy(y_pred, y.repeat(model.num_estimators))

            epoch_loss += loss.item()
            epoch_acc += acc.item()
  
        
    epoch_loss /= len(iterator)
    epoch_acc /= len(iterator)

        
    return epoch_loss, epoch_acc



def evaluate_rank(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0

    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
  
        
    epoch_loss /= len(iterator)
    epoch_acc /= len(iterator)

        
    return epoch_loss, epoch_acc