import torch

def accuracy(y_true, y_pred):
    '''
    Recall that y_pred is not the same as y_logit. y_logit is what comes out of the model, but before
    the final activation. y_pred is the actuall predication. 
    
    y_logit --> pred_prob -->y_pred
    
    How does this look? One possible way is.
    y_logit = model(x)
    pred_prob = torch.softmax(y_logit, dim=1).argmax(dim=1) dim 1 is the column dimension i.e ususally just the array.
    '''
    count = torch.eq(y_true, y_pred).sum().item() #torch eq finds where two arrays are equal
    acc = count/len(y_true)
    
    return acc*100