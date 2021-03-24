'''
Was ich haben will:
automatisch über .fit training und validation output
über .predict fake data generieren
als Daten sollen die Daten reinkommen, für die fake_data generiert werden soll
automatisch die gewünschte Klasse anhängen
zunächst soll angenommen werden, dass die Daten schon richtig skaliert etc sind
zukünftig auch innerhalb der Klasse das Ganze als Dataloader reinpacken
mit normierung und absplitten der gewünschten nachgebauten Klasse
'''
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
import pandas as pd



class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_MSE + loss_KLD

class Autoencoder(nn.Module):
    def __init__(self,D_in=None,H=50,H2=12,latent_dim=3):

        if D_in==None:
            raise ValueError('You need to specify the Input shape.')
        
        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)
        
        # Decoder
        self.linear4=nn.Linear(H2,H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class AutoencoderModel:
    def __init__(self,trainloader,testloader,device,D_in,H=50,H2=12,latent_dim=3):
        self.trainloader=trainloader
        self.testloader=testloader
        self.device=device
        self.D_in=D_in
        self.H=H
        self.H2=H2
        self.latent_dim=latent_dim
        self.model=Autoencoder(D_in, H, H2).to(self.device)
        self.optimizer=optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_mse = customLoss()
    
    def train_model(self,epoch, verbose):
        train_losses = []
        self.model.train()
        train_loss = 0
        for _, data in enumerate(self.trainloader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_mse(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        if verbose:
            if epoch % 200 == 0:        
                print('====> Epoch: {} Average training loss: {:.4f}'.format(
                    epoch, train_loss / len(self.trainloader.dataset)))
                train_losses.append(train_loss / len(self.trainloader.dataset))

    def test_model(self, epoch, verbose):
        test_losses = []
        with torch.no_grad():
            test_loss = 0
            for _, data in enumerate(self.testloader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_mse(recon_batch, data, mu, logvar)
                test_loss += loss.item()
            if verbose:
                if epoch % 200 == 0:        
                    print('====> Epoch: {} Average test loss: {:.4f}'.format(
                        epoch, test_loss / len(self.testloader.dataset)))
                test_losses.append(test_loss / len(self.testloader.dataset))

    def fit(self, epochs, verbose=True):
        for epoch in range(1, epochs + 1):
            self.train_model(epoch, verbose)
            self.test_model(epoch, verbose)
        return self

    def predict(self, no_samples, target_class):
        with torch.no_grad():
            for batch_idx, data in enumerate(self.trainloader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                _, mu_, logvar_ = self.model(data)
                if batch_idx==0:
                    mu=mu_
                    logvar=logvar_
                else:
                    mu=torch.cat((mu, mu_), dim=0)
                    logvar=torch.cat((logvar, logvar_), dim=0)
        sigma = torch.exp(logvar/2)
        no_samples = no_samples
        q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
        z = q.rsample(sample_shape=torch.Size([no_samples]))
        with torch.no_grad():
            pred = self.model.decode(z).cpu().numpy()
        df_fake = pd.DataFrame(pred)
        df_fake['Class']=target_class
        return df_fake

