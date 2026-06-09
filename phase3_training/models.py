import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ResMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(256, 0.15),
            ResidualBlock(256, 0.15),
            ResidualBlock(256, 0.15),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class DeepResMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.1),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(512, 0.15),
            ResidualBlock(512, 0.15),
            ResidualBlock(512, 0.15),
            ResidualBlock(512, 0.15),
            ResidualBlock(512, 0.15),
            ResidualBlock(512, 0.15),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class CustomModel1(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.first_module = nn.Sequential(
            nn.Linear(n_features, 200), nn.BatchNorm1d(200), nn.ReLU(), nn.Dropout(0.2),
        )
        self.second_module = nn.Sequential(
            nn.Linear(200 + n_features, 200), nn.BatchNorm1d(200), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(200, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 50), nn.BatchNorm1d(50), nn.ReLU(), nn.Dropout(0.2),
        )
        self.third_module = nn.Linear(50, 1)

    def forward(self, X):
        first_out = self.first_module(X)
        second_in = torch.cat([X, first_out], dim=1)
        second_out = self.second_module(second_in)
        return self.third_module(second_out)


class MultiBranchModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.spatial_idx = [0, 1, 2, 3, 11]
        self.route_idx = [4, 5, 6, 7, 8, 9, 10]
        self.derived_idx = [12, 13]
        self.temporal_idx = list(range(14, 20))
        self.spatial_branch = nn.Sequential(
            nn.Linear(5, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
        )
        self.route_branch = nn.Sequential(
            nn.Linear(7, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.derived_branch = nn.Sequential(
            nn.Linear(2, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.2),
        )
        self.temporal_branch = nn.Sequential(
            nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(136, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        spatial = self.spatial_branch(x[:, self.spatial_idx])
        route = self.route_branch(x[:, self.route_idx])
        derived = self.derived_branch(x[:, self.derived_idx])
        temporal = self.temporal_branch(x[:, self.temporal_idx])
        combined = torch.cat([spatial, route, derived, temporal], dim=1)
        return self.fusion(combined)


class MultiBranchResNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.osrm_eta_idx = [4]
        self.spatial_idx = [0, 1, 2, 3, 5, 11]
        self.complexity_idx = [6, 7, 8, 9, 10, 12, 13]
        self.temporal_idx = list(range(14, 20))
        self.spatial_branch = nn.Sequential(
            nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            ResidualBlock(32),
        )
        self.complexity_branch = nn.Sequential(
            nn.Linear(7, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            ResidualBlock(64), ResidualBlock(64),
        )
        self.temporal_branch = nn.Sequential(
            nn.Linear(6, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            ResidualBlock(64), ResidualBlock(64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(1 + 32 + 64 + 64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            ResidualBlock(128), ResidualBlock(128),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        osrm_eta = x[:, self.osrm_eta_idx]
        spatial = self.spatial_branch(x[:, self.spatial_idx])
        complexity = self.complexity_branch(x[:, self.complexity_idx])
        temporal = self.temporal_branch(x[:, self.temporal_idx])
        combined = torch.cat([osrm_eta, spatial, complexity, temporal], dim=1)
        return self.fusion(combined)
