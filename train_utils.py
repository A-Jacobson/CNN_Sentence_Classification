from collections import defaultdict
from shutil import copyfile

from tqdm import tqdm_notebook
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model_state, optimizer_state, filename, epoch=None, is_best=False):
    state = dict(model_state=model_state,
                 optimizer_state=optimizer_state,
                 epoch=epoch)
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')


def do_clip(arr, mx):
    clipped = np.clip(arr, (1 - mx) / 1, mx)
    return clipped / clipped.sum(axis=1)[:, np.newaxis]


def binary_accuracy(y_true, y_pred):
    y_pred = y_pred.float()
    y_true = y_true.float()
    return (y_true == torch.round(y_pred)).float().mean()


def categorical_accuracy(y_true, y_pred):
    y_true = y_true.float()
    _, y_pred = torch.max(y_pred, dim=-1)
    return (y_pred.float() == y_true).float().mean()


def _fit_epoch(model, data, criterion, optimizer, batch_size, shuffle):
    model.train()
    running_loss = AverageMeter()
    running_accuracy = AverageMeter()
    loader = DataLoader(data, batch_size, shuffle)
    t = tqdm_notebook(loader, total=len(loader))
    for data, target in t:
        data, target = Variable(data.cuda()), Variable(target.cuda().squeeze())
        output = model(data)
        loss = criterion(output, target)
        accuracy = categorical_accuracy(target.data, output.data)
        running_loss.update(loss.data[0])
        running_accuracy.update(accuracy)
        t.set_description("[ loss: {:.4f} | acc: {:.4f} ] ".format(
            running_loss.avg, running_accuracy.avg))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss.avg, running_accuracy.avg


def fit(model, train, criterion, optimizer, batch_size=32,
        shuffle=True, nb_epoch=1, validation_data=None, cuda=True):
    # TODO: implement CUDA flags, optional metrics and lr scheduler
    if validation_data:
        print('Train on {} samples, Validate on {} samples'.format(len(train), len(validation_data)))
    else:
        print('Train on {} samples'.format(len(train)))

    history = defaultdict(list)
    t = tqdm_notebook(range(nb_epoch), total=nb_epoch)
    for epoch in t:
        loss, acc = _fit_epoch(model, train, criterion,
                              optimizer, batch_size, shuffle)

        history['loss'].append(loss)
        history['acc'].append(acc)
        if validation_data:
            val_loss, val_acc = validate(model, validation_data, criterion, batch_size)
            print("[Epoch {} - loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}]".format(epoch+1,
                                                                                                        loss,
                                                                                                        acc,
                                                                                                        val_loss,
                                                                                                        val_acc))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
        else:
            print("[loss: {:.4f} - acc: {:.4f}]".format(loss, acc))
    return history


def validate(model, validation_data, criterion, batch_size):
    model.eval()
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    for data, target in loader:
        data, target = Variable(data.cuda()), Variable(target.cuda().squeeze())
        output = model(data)
        loss = criterion(output, target)
        accuracy = categorical_accuracy(target.data, output.data)
        val_loss.update(loss.data[0])
        val_accuracy.update(accuracy)
    return val_loss.avg, val_accuracy.avg


def predict(model, data, batch_size):
    model.eval()
    predictions = []
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for data, _ in tqdm_notebook(loader, total=len(loader)):
        data = Variable(data.cuda())
        output = model(data).data
        for prediction in output:
            predictions.append(prediction.cpu().numpy())
    return np.array(predictions)


def predict_labels(model, data, batch_size):
    model.eval()
    labels = []
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for data, _ in tqdm_notebook(loader, total=len(loader)):
        data = Variable(data.cuda())
        _, output = torch.max(model(data).data, dim=-1)
        for label in output:
            labels.append(label.cpu().numpy())
    return np.array(labels)


def generate_predictions(model, data, batch_size):
    model.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for data, _ in loader:
        data = Variable(data.cuda())
        output = model(data).data
        for prediction in output:
            yield prediction.cpu().numpy()