import torch
from torch import nn, optim
from src.dataset import get_loader
from src.models import Generator, Discriminator

EPOCHS, BATCH = 50, 512
NOISE_DIM = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

loader = get_loader(BATCH)
feature_dim = next(iter(loader)).shape[1]

G = Generator(NOISE_DIM, feature_dim).to(device)
D = Discriminator(feature_dim).to(device)

opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

for epoch in range(1, EPOCHS + 1):
    for real in loader:
        real = real.to(device)
        real += 0.05 * torch.randn_like(real)  # Input noise

        size = real.size(0)
        noise = torch.randn(size, NOISE_DIM, device=device)

        # Train Discriminator (2 steps)
        for _ in range(2):
            fake = G(noise).detach()
            D_real = D(real).view(-1)
            D_fake = D(fake).view(-1)

            real_labels = torch.full_like(D_real, 0.9)  # label smoothing
            loss_D = criterion(D_real, real_labels) + criterion(D_fake, torch.zeros_like(D_fake))

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # Train Generator
        fake = G(noise)
        output = D(fake).view(-1)
        loss_G = criterion(output, torch.ones_like(output))

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"{epoch}/{EPOCHS} | D: {loss_D.item():.3f} | G: {loss_G.item():.3f}")

torch.save(G.state_dict(), "G.pt")
torch.save(D.state_dict(), "D.pt")
