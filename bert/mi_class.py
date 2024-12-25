import numpy as np
import math

import torch
import torch.nn as nn
import sys
# sys.path.append('/home/sunmingchen/liyingji/CDDD/debias-BERT')
from bert.info_nce import InfoNCE as infonce


class FilterMU_Unite(nn.Module):
    def __init__(self, filter_input_size, filter_hidden_size, mu_hidden_size, mode="info"):  # or ["info","sim"]
        super(FilterMU_Unite, self).__init__()
        self.loss = None
        self.encoder = nn.Sequential(
                            nn.Linear(filter_input_size, filter_hidden_size),
                            nn.ReLU()
                            #nn.Linear(filter_hidden_size, filter_hidden_size))
                            )
        self.decoder_a = nn.Linear(filter_hidden_size, filter_input_size)
        self.decoder_b = nn.Linear(filter_hidden_size, filter_input_size)
        self.mode = mode
        if self.mode == 'club_infonce':
            self.infonce = InfoNCE(filter_hidden_size, filter_input_size, mu_hidden_size)
        elif self.mode == 'info':
            self.infonce = infonce()
        self.recon_loss = nn.MSELoss()

    def encode(self,sent_embedding):
        out = self.encoder(sent_embedding)
        return out

    def forward(self, sent_emb_a, word_emb_a, sent_emb_b, word_emb_b, lam, club=None):
        debias_emb_a = self.encode(sent_emb_a)
        debias_emb_b = self.encode(sent_emb_b)
        if self.mode == "club_infonce":
            self.loss2 = self.infonce.mi_est(debias_emb_a, debias_emb_b)
            debias_emb = torch.cat([debias_emb_a, debias_emb_b], 0)
            word_emb = torch.cat([word_emb_a, word_emb_b], 0)
            self.loss1 = club.mi_est(debias_emb, word_emb)
            self.loss = - self.loss2 + lam * self.loss1

        elif self.mode == "info":
            # self.loss = self.infonce.mi_est(debias_emb_a, debias_emb_b)
            self.loss = self.infonce(debias_emb_a, debias_emb_b)
        return self.loss


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            mi_est() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''
    def __init__(self, x_dim, y_dim):
        super(CLUB, self).__init__()
        self.p_mu = nn.Linear(x_dim,y_dim)
        self.p_logvar = nn.Linear(x_dim,y_dim)

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def mi_est(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()

        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp()

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())

    def mi_est(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size))
        return lower_bound
