import torch
from torch import nn
from typing import Tuple
import numpy as np
import random
import copy

# =========================
# Utils
# =========================

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

# =========================
# Inception Block
# =========================

class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_sizes: Tuple[int, int, int] = (40, 20, 10),
        bottleneck: bool = True,
    ):
        super().__init__()

        self.use_bottleneck = bottleneck and in_channels > 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, nb_filters, kernel_size=1, bias=False
            )
            conv_in = nb_filters
        else:
            self.bottleneck = nn.Identity()
            conv_in = in_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(
                conv_in,
                nb_filters,
                kernel_size=k,
                padding="same",
                bias=False,
            )
            for k in kernel_sizes
        ])

        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
        )

        self.activation = nn.ReLU()

        # saída sempre = 4 * nb_filters
        self.out_channels = 4 * nb_filters

    def forward(self, x):
        x_b = self.bottleneck(x)

        outputs = [conv(x_b) for conv in self.convs]
        outputs.append(self.maxpool_branch(x))

        x = torch.cat(outputs, dim=1)
        return self.activation(x)

# =========================
# Single Hinception Model (Regressor)
# =========================

class Hinception(nn.Module):
    def __init__(self, sequence_length, in_channels, output_dim=1, nb_filters=32, hname="H1"):
        super().__init__()

        self.hname = hname

        # =====================
        # Custom Filters
        # =====================

        custom_kernels = [2, 4, 8, 16, 32, 64]
        custom_convs = []

        for ks in custom_kernels:
            filt = np.ones((1, in_channels, ks))
            filt[:, :, np.arange(ks) % 2 == 0] *= -1
            conv = nn.Conv1d(in_channels, 1, ks, padding="same", bias=False)
            conv.weight = nn.Parameter(torch.from_numpy(filt).float(), requires_grad=False)
            custom_convs.append(conv)

        for ks in custom_kernels:
            filt = np.ones((1, in_channels, ks))
            filt[:, :, np.arange(ks) % 2 > 0] *= -1
            conv = nn.Conv1d(in_channels, 1, ks, padding="same", bias=False)
            conv.weight = nn.Parameter(torch.from_numpy(filt).float(), requires_grad=False)
            custom_convs.append(conv)

        self.custom_convs = nn.ModuleList(custom_convs)
        self.custom_out_channels = len(custom_convs)  # 12
        self.custom_activation = nn.ReLU()

        # =====================
        # Inception Backbone
        # =====================

        self.inception1 = InceptionModule(in_channels, nb_filters)
        in_ch_2 = self.inception1.out_channels + self.custom_out_channels

        self.inception2 = InceptionModule(in_ch_2, nb_filters)
        self.inception3 = InceptionModule(self.inception2.out_channels, nb_filters)

        self.inception4 = InceptionModule(self.inception3.out_channels, nb_filters)
        self.inception5 = InceptionModule(self.inception4.out_channels, nb_filters)
        self.inception6 = InceptionModule(self.inception5.out_channels, nb_filters)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(self.inception6.out_channels, output_dim)

    def forward(self, x):

        # Custom maps
        custom_maps = torch.cat([conv(x) for conv in self.custom_convs], dim=1)
        custom_maps = self.custom_activation(custom_maps)

        # Inception pipeline
        f1 = self.inception1(x)
        f1 = torch.cat([f1, custom_maps], dim=1)

        f2 = self.inception2(f1)
        f3 = self.inception3(f2)

        f4 = self.inception4(f3)
        f5 = self.inception5(f4)
        f6 = self.inception6(f5)

        features = self.global_pool(f6).squeeze(-1)
        return self.linear(features)

    # =====================
    # Training (Regression)
    # =====================

    def to_train(self, loader, loss_fn, opt, device):
        self.train()
        self.to(device)

        total_loss = 0.0
        total_samples = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            preds = self(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader, loss_fn, device):
        self.eval()
        self.to(device)

        total_loss = 0.0
        total_samples = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = self(x)
            loss = loss_fn(preds, y)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        return total_loss / total_samples

# =========================
# Ensemble
# =========================

class HinceptionTime(nn.Module):
    def __init__(
        self,
        configs,
        sequence_length,
        in_channels,
        output_dim=1,
        num_models=5,
        seed=42,
    ):
        super().__init__()

        self.models = nn.ModuleList()

        for i in range(num_models):
            set_seed(seed * 100 + i)
            self.models.append(
                Hinception(sequence_length, in_channels, output_dim, hname=f"H{i+1}")
            )

        self.losses = [configs["loss"] for _ in range(num_models)]
        self.opts = [
            configs["optimizer"](m.parameters(), configs["lr"])
            for m in self.models
        ]

    def forward(self, x):
        outputs = torch.stack([m(x) for m in self.models], dim=1)
        return torch.mean(outputs, dim=1)

    def run(self, train_loader, epochs, device, patience=50):

        for model, loss_fn, opt in zip(
            self.models, self.losses, self.opts
        ):

            print(f"\nTraining {model.hname}\n")

            best_loss = float("inf")
            best_model_state = None
            epochs_no_improve = 0

            for epoch in range(epochs):

                train_loss = model.to_train(train_loader, loss_fn, opt, device)

                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train MSE: {train_loss:.6f}"
                )

                if train_loss < best_loss:
                    best_loss = train_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1}. "
                        f"Best training loss: {best_loss:.6f}"
                    )
                    break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)