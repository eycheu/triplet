import torch
import numpy as np

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Arguments:
    train_loader: DataLoader: Loader for training data.
    val_loader: DataLoader: Loader for testing data (default: None).
    model: Module
    loss_fn: Module
    optimizer: Optimizer:
    scheduler: 
    n_epochs: Integer:
    cuda: Bool:
    log_interval: Integer:
    metrics: List: 
    """
    if scheduler is not None:
        for epoch in range(0, start_epoch):
            scheduler.step()

    dict_train_loss = dict()
    dict_val_loss = dict()
    for epoch in range(start_epoch, n_epochs):
        if scheduler is not None:
            scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
        dict_train_loss[epoch+1] = train_loss
        
        msg = 'Epoch [{}/{}]: Avg. Train Loss: {:.5f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            msg += ' | {}: {:.5f}'.format(metric.name(), metric.value())
        
        if val_loader is not None:
            val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
            dict_val_loss[epoch+1] = val_loss
            
            val_loss /= len(val_loader)
            msg += '\nEpoch [{}/{}]: Avg. Validation Loss: {:.5f}'.format(epoch + 1, n_epochs, val_loss)
            for metric in metrics:
                msg += ' | {}: {:.5f}'.format(metric.name(), metric.value())
            
        print(msg)
        
    return dict_train_loss, dict_val_loss

def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()
    
    n_batches = len(train_loader)
   
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if log_interval is not None and (batch_idx % log_interval == 0 or batch_idx == (n_batches - 1)):
            msg = 'Batch [{}/{} ({:.1f}%)]: Loss: {:>.5f}'.format(
                batch_idx + 1, n_batches,
                100. * (batch_idx + 1) / n_batches, 
                np.mean(losses))
            for metric in metrics:
                msg += ' | {}: {:.5f}'.format(metric.name(), metric.value())

            print(msg)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
