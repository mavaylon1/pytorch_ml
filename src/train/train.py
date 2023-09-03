import torch
from torch import nn
from timeit import default_timer as timer
from tqdm import tqdm

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device,
          accuracy_fn):
    """
    This functions acts as the operation each epoch in the training.
    """
    model = model
    loss_fn = loss_fn
    optimizer = optimizer
    accruacy_fn = accuracy_fn
    device = device
    dataloader = dataloader
    
    # move model to device and set to train mode
    model.to(device)
    model.train()
    
    # The loss and accuracy are the respective average over the batches.
    epoch_loss = 0
    epoch_accuracy = 0
    
    # conduct training over batches
    for X,y in dataloader: # each iteration is a batch
        # move data to device
        X = X.to(device)
        y = y.to(device)
        
        # get model output
        y_logit = model(X)
        
        # calculate loss and accuracy
        loss = loss_fn(y_logit, y)
        
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # argmax will return the index of the largest prob, which in this case is the class prediction
        accuracy = accuracy_fn(y, y_pred) # recall y_logit -> y_pred
        
        epoch_loss += loss
        epoch_accuracy += accuracy
        
        # zero gradients on the optimizer
        optimizer.zero_grad()
        
        # perform backprop on loss via backwards
        loss.backward()
        
        # update gradients via optimizer
        optimizer.step()
    
    epoch_loss = epoch_loss/len(dataloader)
    epoch_accuracy = epoch_accuracy/len(dataloader)
    
    print(f'Training Epoch Loss: {epoch_loss}, Training Epoch Accuracy: {epoch_accuracy}')

    
def validation(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          device,
          accuracy_fn):
    """
    This functions acts as the validation step for each epoch in the training.
    """
    model = model
    loss_fn = loss_fn
    accruacy_fn = accuracy_fn
    device = device
    dataloader = dataloader
    
    # move model to device
    model.to(device)
    
    # set model in eval mode
    model.eval()
    
    # The loss and accuracy are the respective average over the batches.
    epoch_loss = 0
    epoch_accuracy = 0
    
    # validate over batches
    with torch.inference_mode(): 
        for X,y in dataloader:
            # move data to device
            X = X.to(device)
            y = y.to(device)

            # get model output
            y_logit = model(X)

            # calculate loss and accuracy
            loss = loss_fn(y_logit, y)

            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # argmax will return the index of the largest prob, which in this case is the class prediction
            accuracy = accuracy_fn(y, y_pred) # recall y_logit -> y_pred

            epoch_loss += loss
            epoch_accuracy += accuracy

        epoch_loss = epoch_loss/len(dataloader)
        epoch_accuracy = epoch_accuracy/len(dataloader)

        print(f'Validation Epoch Loss: {epoch_loss}, Validation Epoch Accuracy: {epoch_accuracy}')
    
    
def train_manager(model: torch.nn.Module,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device,
                  accuracy_fn,
                  epochs: int,
                  train_dataloader,
                  validation_step: bool = False,
                  val_dataloader = None):
    """
    This manages model training and validation for user defined epochs. 
    This reports loss, accuracy, and training time.
    
    Validation is optional
    """
    epochs = epochs
    model = model
    loss_fn = loss_fn
    accruacy_fn = accuracy_fn
    device = device
    validation_step = validation_step
    train_dataloader = train_dataloader
    val_dataloader = val_dataloader
    optimizer = optimizer
    
    train_time_start = timer()
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train(dataloader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device
        )
        
        if validation_step:
            if val_dataloader is None:
                msg = "Validation is set to True. Please provide validation dataloader or change validation to False"
                raise ValueError(msg)
            
            else:
                validation(dataloader=val_dataloader,
                model=model,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                device=device
            )

    train_time_end = timer()
    total_train_time_model_2 = print_train_time(start=train_time_start,
                                               end=train_time_end,
                                               device=device)