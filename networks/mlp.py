import torch
import torch.nn as nn


class MyEnc(nn.Module):
    def __init__(self, enc_dims):
        super(MyEnc, self).__init__()
        stack = len(enc_dims) - 1
        self.enc_module = []
        for i in range(stack - 1):
            self.enc_module.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            self.enc_module.append(nn.ReLU())
        self.enc_module.append(nn.Linear(enc_dims[-2], enc_dims[-1]))
        self.encoder = nn.Sequential(*self.enc_module)
        print(self.encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            self.fc1,
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MySupConEnc(nn.Module):
    def __init__(self, enc_dims):
        super(MySupConEnc, self).__init__()
        if enc_dims is None:
            enc_dims = [1024, 512, 256]

        self.encoder = MyEnc(enc_dims)
        self.feature_dim = enc_dims[-1]

        # projection head
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

    def forward(self, x, return_feat=False):
        encoded = self.encoder(x)
        feat = self.head(encoded)
        if return_feat:
            return feat, encoded
        else:
            return feat


class SupConMLP(nn.Module):
    def __init__(self, feat_dim=500):
        super(SupConMLP, self).__init__()
        self.encoder = MLP()
        self.head = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500)
        )

    def forward(self, x, return_feat=False):
        encoded = self.encoder(x)
        feat = self.head(encoded)
        if return_feat:
            return feat, encoded
        else:
            return feat


class MyBinaryClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dims=None):
        super(MyBinaryClassifier, self).__init__()

        self.num_classes = 2
        if hidden_dims is None:
            hidden_dims = [100, 100]
        stack = len(hidden_dims) - 1

        self.mlp_module = []
        self.mlp_module.append(nn.Linear(feature_dim, hidden_dims[0]))
        self.mlp_module.append(nn.ReLU())

        for i in range(stack):
            self.mlp_module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.mlp_module.append(nn.ReLU())

        self.mlp_module.append(nn.Linear(hidden_dims[-1], self.num_classes))
        self.mlp_modules.append(nn.Softmax(dim=1))

        self.fc = nn.Sequential(*self.mlp_module)

    def forward(self, features):
        return self.fc(features)


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, num_classes=10, feature_dim=None):
        super(LinearClassifier, self).__init__()
        if feature_dim is None:
            feature_dim = 500
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
