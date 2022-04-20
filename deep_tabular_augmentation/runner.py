from .common import *
from .callbacks import *

from typing import Any
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


class Learner():
    """ helper class in which all the data relevant part is stored
    """
    def __init__(self, model, opt, loss_func, data:DataLoader, target_name:str=None, target_class:int=None, cols:list=None, cont_vars:list=None):
        self.model,self.opt,self.loss_func,self.data,self.target_name,self.target_class,self.cols,self.cont_vars = model,opt,loss_func,data,target_name,target_class,cols,cont_vars


class Runner():
    """ helper class which acts as a pipeline through which the data should run
    """
    def __init__(self, cbs=None, cb_funcs=None):
        self.in_train = False
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    @property
    def opt(self):              return self.learn.opt
    @property
    def model(self):            return self.learn.model
    @property
    def loss_func(self):        return self.learn.loss_func
    @property
    def data(self):             return self.learn.data
    @property
    def target_name(self):      return self.learn.target_name
    @property
    def target_class(self):     return self.learn.target_class
    @property
    def cols(self):             return self.learn.cols
    @property
    def cont_vars(self):        return self.learn.cont_vars

    def one_batch(self, xb, _):
        try:
            self.xb = xb
            self('begin_batch')
            self.recon_batch, self.mu, self.logvar = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.recon_batch, self.xb, self.mu, self.logvar)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb,_ in dl: self.one_batch(xb, _)
        except CancelEpochException: self('after_cancel_epoch')

    def fit(self, epochs, learn):
        self.epochs,self.learn,self.loss = epochs,learn,torch.tensor(0.)

        try:
            for cb in self.cbs: cb.set_runner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                with torch.no_grad():
                    if not self('begin_validate'): self.all_batches(self.data.test_dl)
                self('after_epoch')

        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.learn = None

    def _get_embeddings(self):
        self.in_train = False
        with torch.no_grad():
            for batch_idx, (xb,_) in enumerate(self.data.train_dl):
                self.opt.zero_grad()
                _, mu_, logvar_ = self.model(xb)
                if batch_idx==0:
                    mu=mu_
                    logvar=logvar_
                else:
                    mu=torch.cat((mu, mu_), dim=0)
                    logvar=torch.cat((logvar, logvar_), dim=0)
        return mu, logvar


    def predict_df(self, learn, no_samples:int, scaler=None):
        self.learn = learn
        mu, logvar = self._get_embeddings()

        sigma = torch.exp(logvar/2)
        no_samples = no_samples
        q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
        z = q.rsample(sample_shape=torch.Size([no_samples]))

        with torch.no_grad():
            pred = self.model.decode(z).cpu().numpy()

        if self.target_name:
            if self.target_name in self.cols: self.cols.remove(self.target_name)
        df_fake = pd.DataFrame(pred, columns=self.cols)
        
        if scaler:
            if self.cont_vars:
                df_fake[self.cont_vars]=scaler.inverse_transform(df_fake[self.cont_vars])
            else:
                df_fake=pd.DataFrame(scaler.inverse_transform(df_fake), columns=self.cols)

        if self.target_name: df_fake[self.target_name]=self.target_class

        return df_fake

    def predict_with_noise_df(self, learn, no_samples:int, mu:float, sigma:list, scaler=None):
        self.learn = learn

        df_fake_with_noise = self.predict_df(learn, no_samples, scaler=scaler)
        np_matrix = df_fake_with_noise[self.cols].values
        np_matrix = np.array([val+np.random.normal(mu, sigma[i], 1) for sublist in np_matrix for i, val in enumerate(sublist)]).reshape(-1,np_matrix.shape[1])
        df_fake_with_noise = pd.DataFrame(np_matrix, columns=self.cols)
        if self.target_name: df_fake_with_noise[self.target_name]=self.target_class

        return df_fake_with_noise

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) or res
        return res