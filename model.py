import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)

class VAE(pl.LightningModule):
    def __init__(self, is_mnist, enc_out_dim=512, latent_dim=256, input_height=32, num_embed=10):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim, 
            input_height=input_height, 
            first_conv=False, 
            maxpool1=False
        )

        if is_mnist:
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.decoder.conv1 = nn.Conv2d(64*self.decoder.expansion, 1, kernel_size=3, stride=1, padding=3, bias=False)

        # distribution parameters
        self.img_fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.img_fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.label_embed = nn.Embedding(num_embed, enc_out_dim)

        # distribution parameters
        self.lbl_fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.lbl_fc_var = nn.Linear(enc_out_dim, latent_dim)

        self.experts = ProductOfExperts()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, batch):
        x, y = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x) 
        y_encoded = self.label_embed(y)

        img_mu, img_log_var = self.img_fc_mu(x_encoded), self.img_fc_var(x_encoded)
        lbl_mu, lbl_log_var = self.lbl_fc_mu(y_encoded), self.lbl_fc_var(y_encoded)
        
        mu = torch.cat((img_mu.unsqueeze(0), lbl_mu.unsqueeze(0)), dim=0)
        log_var = torch.cat((img_log_var.unsqueeze(0), lbl_log_var.unsqueeze(0)), dim=0)

        mu, log_var = self.experts(mu, log_var)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded 
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x) 
        y_encoded = self.label_embed(y)

        img_mu, img_log_var = self.img_fc_mu(x_encoded), self.img_fc_var(x_encoded)
        lbl_mu, lbl_log_var = self.lbl_fc_mu(y_encoded), self.lbl_fc_var(y_encoded)
        
        mu = torch.cat((img_mu.unsqueeze(0), lbl_mu.unsqueeze(0)), dim=0)
        log_var = torch.cat((img_log_var.unsqueeze(0), lbl_log_var.unsqueeze(0)), dim=0)

        mu, log_var = self.experts(mu, log_var)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded 
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(), 
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo
    
class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar