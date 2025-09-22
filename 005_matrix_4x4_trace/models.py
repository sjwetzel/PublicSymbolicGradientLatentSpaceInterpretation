import torch

class SimpleLinear(torch.nn.Module):
    def __init__(self,
                 in_feats: int,
                 out_feats: int):
        super(SimpleLinear, self).__init__()

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(in_feats, 1000),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(1000, 1000),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(1000, 1),
        )

        self.activation = torch.nn.Sigmoid()


    def forward(self, x):
        latent = self.latent(x)
        return self.activation(latent), latent

class LinearFFW(torch.nn.Module):
    def __init__(self,
                 in_feats: int,
                 hidden_feats: int,
                 out_feats: int):
        super(LinearFFW, self).__init__()

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_feats, hidden_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_feats, hidden_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_feats, hidden_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_feats, out_feats),
        )

    def forward(self, x):
        return self.linear(x)

class SiameseNetwork(torch.nn.Module):
    def __init__(self,
                 in_feats: int = 1,
                 hidden_feats: int = 160,
                 out_feats: int = 1):
        
        super(SiameseNetwork, self).__init__()

        self.shared_dense = LinearFFW(in_feats, hidden_feats, out_feats)

    def forward(self, x):

        x_a, x_p, x_n = x.chunk(3, dim=1)

        latent_a = self.shared_dense(x_a.squeeze())
        latent_p = self.shared_dense(x_p.squeeze())
        latent_n = self.shared_dense(x_n.squeeze())

        return latent_a, latent_p, latent_n