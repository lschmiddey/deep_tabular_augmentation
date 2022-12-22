from .common import *

from typing import Any
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from functools import partial
import math



class Callback():
    _order=0
    def set_runner(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs=0.
        self.run.n_iter=0

    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1

    def begin_epoch(self):
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False


class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass


class LossTracker(Callback):
    def __init__(self, show_every): self.show_every = show_every

    def begin_fit(self):
        self.train_losses = []
        self.val_losses = []
    
    def after_batch(self):
        if self.in_train:
            self.train_losses.append(self.loss.detach().cpu()) 
        else:
            self.val_losses.append(self.loss.detach().cpu())

    def after_epoch(self):
        if self.run.epoch % self.show_every==0:
            print(f'epoch: {self.run.epoch + self.show_every}')
            print(f'train loss is: {sum(self.train_losses)/len(self.train_losses)}')
            print(f'validation loss is: {sum(self.val_losses)/len(self.val_losses)}')

    def plot_train_vs_val_loss(self, skip_last=0):
        figure(figsize=(8, 6), dpi=80)
        plt.plot(self.train_losses[:len(self.train_losses)-skip_last], label="train data")
        plt.plot(self.val_losses[:len(self.val_losses)-skip_last], label="validation data")
        plt.legend()
        plt.show()

    def plot_train_vs_val_loss_last_epochs(self, show_last=100):
        figure(figsize=(8, 6), dpi=80)
        plt.plot(self.train_losses[len(self.train_losses)-show_last:], label="train data")
        plt.plot(self.val_losses[len(self.val_losses)-show_last:], label="validation data")
        plt.legend()
        plt.show()


class Recorder(Callback):
    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []
        self.losses_val = []

    def after_batch(self):
        if not self.in_train: return
        for pg,lr in zip(self.opt.param_groups,self.lrs): lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])
    def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])

    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs    = self.lrs[pgid]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])


class ParamScheduler(Callback):
    _order=1
    def __init__(self, pname, sched_funcs): self.pname,self.sched_funcs = pname,sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list,tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups)==len(self.sched_funcs)
        for pg,f in zip(self.opt.param_groups,self.sched_funcs):
            pg[self.pname] = f(self.n_epochs/self.epochs)

    def begin_batch(self):
        if self.in_train: self.set_param()


def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner


@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2


def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        if idx == 2: idx = 1
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner

class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.param_groups: pg['lr'] = lr

    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss
