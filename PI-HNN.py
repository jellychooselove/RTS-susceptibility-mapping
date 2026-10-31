
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PI_HNN_MechOnly(nn.Module):
    """
    Physics-Informed Hybrid Neural Network
    Mechanics-only configuration for RTS susceptibility.
    """

    def __init__(self, num_cont, cat_dims, embed_dim=8):
        super().__init__()

        # 1. Categorical feature embeddings
        self.emb_soil = nn.Embedding(cat_dims["soil_type"], embed_dim)
        self.emb_permafrost = nn.Embedding(
            cat_dims["permafrost_type"], embed_dim
        )
        self.emb_geomor = nn.Embedding(
            cat_dims["geomor_type"], embed_dim
        )

        # 2. Continuous feature encoder
        self.cont_net = nn.Sequential(
            nn.Linear(num_cont, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # 3. Feature fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 3 * embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 4. Mechanical parameter estimation
        self.cohesion_head = nn.Linear(64, 1)
        self.friction_head = nn.Linear(64, 1)
        self.pore_pressure_head = nn.Linear(64, 1)

        # 5. Physics-guided gating network
        self.phys_gate_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )

        # 6. Residual prediction network
        self.residual_head = nn.Sequential(
            nn.Linear(68, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        # Learnable uncertainty-related loss parameters
        self.log_var_data = nn.Parameter(torch.zeros(1))
        self.log_var_phys = nn.Parameter(torch.zeros(1))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_cont, x_soil, x_permafrost,
                x_geomor, phys_inputs):

        # Feature encoding and fusion
        emb = torch.cat([
            self.emb_soil(x_soil),
            self.emb_permafrost(x_permafrost),
            self.emb_geomor(x_geomor)
        ], dim=1)

        shared_features = self.fusion_net(
            torch.cat([self.cont_net(x_cont), emb], dim=1)
        )

        # Mechanical parameter estimation
        c = 10.0 + 50.0 * torch.sigmoid(
            self.cohesion_head(shared_features).squeeze(-1)
        )

        phi = (
            15.0 + 30.0 * torch.sigmoid(
                self.friction_head(shared_features).squeeze(-1)
            )
        ) * (math.pi / 180.0)

        u_w = 20.0 * torch.sigmoid(
            self.pore_pressure_head(shared_features).squeeze(-1)
        )

        # Infinite-slope mechanical formulation
        alpha = phys_inputs["slope_deg"] * (math.pi / 180.0)
        h = torch.clamp(
            phys_inputs["permafrost_depth"], min=0.5
        )
        gamma_g_h = (
            phys_inputs["soil_bulk_density"] * 9.81 * h
        )

        tau_driving = (
            gamma_g_h * torch.sin(alpha) * torch.cos(alpha)
        ) / 100.0

        sigma_eff = (
            gamma_g_h * torch.cos(alpha) ** 2
        ) / 100.0 - u_w

        tau_resisting = (
            c + torch.clamp(sigma_eff, min=0.1) * torch.tan(phi)
        )

        Fs = torch.clamp(
            tau_resisting / (tau_driving + 1e-6),
            min=0.1, max=10.0
        )

        u_ratio = u_w / (gamma_g_h / 100.0 + 1e-6)

        # Physics-guided feature modulation
        phys_feat = torch.stack([
            Fs, tau_driving, sigma_eff, u_ratio
        ], dim=1)

        gate = self.phys_gate_net(phys_feat)
        gated_features = shared_features * gate

        # Residual susceptibility prediction
        res_input = torch.cat(
            [gated_features, phys_feat], dim=1
        )

        phys_bias = 0.2 * (1.1 - Fs)

        logit = (
            self.residual_head(res_input).squeeze(-1)
            + phys_bias
        )

        probability = torch.sigmoid(logit)
        probability = torch.clamp(
            probability, 1e-7, 1.0 - 1e-7
        )

        return probability, c, phi, u_w, Fs


class MechanicalConsistencyLoss:

    @staticmethod
    def loss(p, c, phi, u_w, inputs):

        alpha = torch.clamp(
            inputs["slope_deg"] * np.pi / 180.0,
            min=0.05
        )

        h = torch.clamp(
            inputs["permafrost_depth"], min=0.1
        )

        rho = inputs["soil_bulk_density"] * 10.0
        g = 9.81

        sigma_total = (
            rho * g * h * torch.cos(alpha) ** 2
        ) / 1000.0

        tau_driving = (
            rho * g * h * torch.sin(alpha)
            * torch.cos(alpha)
        ) / 1000.0

        sigma_eff = torch.clamp(
            sigma_total - u_w, min=0.5
        )

        tau_resisting = (
            c + sigma_eff * torch.tan(phi)
        )

        Fs = tau_resisting / (tau_driving + 1e-6)

        # Smooth mechanical susceptibility target
        p_phys_target = torch.sigmoid(
            1.5 * (1.0 - Fs)
        )

        consistency_loss = F.mse_loss(
            p, p_phys_target
        )

        # Mechanical parameter regularization
        reg = 0.005 * torch.mean(
            (c - 20.0) ** 2 + (phi - 0.5) ** 2
        )

        return consistency_loss + reg
