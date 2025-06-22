# app.py

import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Model Definition ---
class CVAE(torch.nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.fc1 = torch.nn.Linear(784 + 10, 400)
        self.fc21 = torch.nn.Linear(400, latent_dim)
        self.fc22 = torch.nn.Linear(400, latent_dim)
        self.fc3 = torch.nn.Linear(latent_dim + 10, 400)
        self.fc4 = torch.nn.Linear(400, 784)

    def encode(self, x, y):
        h1 = torch.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = torch.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h3))

# --- Load Model ---
device = torch.device("cpu")
model = CVAE().to(device)
model.load_state_dict(torch.load("train_cvae_mnist.ipynb", map_location=device))
model.eval()

# --- UI ---
st.title("🧠 Handwritten Digit Generator")
digit = st.selectbox("Choose a digit (0–9):", list(range(10)))

if st.button("Generate Images"):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        z = torch.randn(1, 20)
        y = torch.zeros(1, 10)
        y[0][digit] = 1

        with torch.no_grad():
            generated = model.decode(z, y).view(28, 28).numpy()

        axs[i].imshow(generated, cmap="gray")
        axs[i].axis("off")

    st.pyplot(fig)
