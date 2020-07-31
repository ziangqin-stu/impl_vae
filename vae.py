import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torchvision import datasets, transforms
import wandb
from utils import plot_image_plane

class Encoder(torch.nn.Module):
    # see appendix C.2
    def __init__(self, x_dim, z_dim, h_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.Tanh()
        )
        self.linear_mu = nn.Linear(h_dim, z_dim)
        self.linear_sigma = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        h = self.linear(x)
        mu = self.linear_mu(h)
        log_sigma_sq = self.linear_sigma(h)
        sigma_sq = torch.exp(log_sigma_sq)
        return mu, sigma_sq


class Decoder(torch.nn.Module):
    # see appendix C.2
    def __init__(self, x_dim, z_dim, h_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh()
        )
        self.linear_mu = nn.Linear(h_dim, x_dim)
        self.linear_sigma = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        h = self.linear(z)
        mu = self.linear_mu(h)
        log_sigma_sq = self.linear_sigma(h)
        sigma_sq = torch.exp(log_sigma_sq)
        return mu, sigma_sq


class SimpleDecoder(torch.nn.Module):
    # see appendix C.2
    def __init__(self, x_dim, z_dim, h_dim):
        super(SimpleDecoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.linear(z)
        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder, x_dim, z_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.z_mu = None
        self.z_sigma_sq = None
        self.x_mu = None
        self.x_sigma_sq = None

    def _sample_latent(self, mu, sigma):
        # reparameterization trick, see 2.4
        epsilon = torch.randn(self.z_dim)
        z = mu + sigma * epsilon
        return z

    def _sample_reconstruction(self, mu, sigma):
        # similar to with _sample_latent
        epsilon = torch.randn(self.x_dim)
        x_ = mu + sigma * epsilon
        return x_

    def forward(self, x):
        mu, sigma_sq = self.encoder(x)
        z = self._sample_latent(mu, torch.sqrt(sigma_sq))
        mu_, sigma_sq_ = self.decoder(z)
        x_ = self._sample_reconstruction(mu_, torch.sqrt(sigma_sq_))
        self.z_mu = mu
        self.z_sigma_sq = sigma_sq
        self.x_mu = mu_
        self.x_sigma_sq = sigma_sq_
        return x_

    def forward_simple(self, x):
        mu, sigma_sq = self.encoder(x)
        z = self._sample_latent(mu, torch.sqrt(sigma_sq))
        x_ = self.decoder(z)
        self.z_mu = mu
        self.z_sigma_sq = sigma_sq
        return x_

    def sample(self):
        # sample from standard Gaussian (as in assumption)
        z_ = torch.randn(9, self.z_dim)
        x_ = self.decoder(z_)
        return x_

def latent_loss(mu, sigma_sq):
    # see appendix F.1
    mu_sq = mu * mu
    lbo_phi = .5 * torch.mean(1 + torch.log(sigma_sq + 1e-10) - mu_sq - sigma_sq)
    lbo_phi = torch.clamp(lbo_phi, -10000,  10000)
    return -lbo_phi


if __name__ == '__main__':
    # hyper parameters, see 5 and 2.3
    x_dim = 28 * 28
    z_dim = 10
    h_dim = 500
    batch_size = 32
    epoch_num = int(200)
    seed = 0
    lr = 1e-4
    beta = 0.001

    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST('./data/mnist', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)
    wandb.init(project="ziang-vae")
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(x_dim, z_dim, h_dim).to(device)
    decoder = Decoder(x_dim, z_dim, h_dim).to(device)
    vae = VAE(encoder, decoder, x_dim, z_dim).to(device)
    loss_rec_func = nn.MSELoss()
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)


    print("Start Training ({} epochs, seed={}, lr={}, beta={})...".format(epoch_num, seed, lr, beta))
    print("-------------------------------------------------------------")
    for epoch in range(epoch_num):
        for i, data in enumerate(dataloader, 0):
            x, _ = data
            x = Tensor(x.resize_(batch_size, x_dim)).to(device)
            x_ = vae.forward(x)
            loss_latent_encoder = latent_loss(vae.z_mu, vae.z_sigma_sq)
            loss_latent_decoder = latent_loss(vae.x_mu, vae.x_sigma_sq)
            loss = beta * loss_latent_encoder + loss_latent_decoder
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch + 1, loss.data.item(), loss_latent_encoder.data.item(), loss_latent_decoder.data.item())
        wandb.log({'loss': loss.data})
        wandb.log({'latent encoder loss': loss_latent_encoder.data})
        wandb.log({'latent decoder loss': loss_latent_decoder.data})
        wandb.log({"reconstructed image": [wandb.Image(plot_image_plane(x_.detach().cpu(), 3,3), caption="recn_epoch={}".format(epoch))]})
        wandb.log({"sampled image": [wandb.Image(plot_image_plane(vae.sample().detach().cpu(), 3,3), caption="samp_epoch={}".format(epoch))]})
    print("-------------------------------------------------------------")
    print("Training Finished.".format(epoch_num))
