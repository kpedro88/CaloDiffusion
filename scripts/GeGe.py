import torch
import torch.nn as nn
import numpy as np

# based on https://github.com/sprillo/softsort
# (S. Prillo, J. M. Eisenschlos, "SoftSort: A Continuous Relaxation for the argsort Operator", https://arxiv.org/abs/2006.16038, ICML 2020)
# differentiable exact sort: calculates orthonormal matrix from "soft" scores using only operations w/ gradients
class HardSort(torch.nn.Module):
    def __init__(self, tau=1.0, pow=1.0):
        super().__init__()
        # do these hyperparameters matter in practice?
        self.tau = tau
        self.pow = pow

    def forward(self, scores: torch.Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        sorted = scores.sort(descending=True, dim=1)[0]
        P_hat = None
        # build up a sparse representation of pairwise_diff without realizing full NxN dense matrix in memory
        for row in range(sorted.size()[1]):
            P_tmp = (scores - sorted[:,row].unsqueeze(-1)).unsqueeze(-2).abs().pow(self.pow).neg() / self.tau
            # all values less than max go below zero, then send negatives to zero w/ ReLU
            P_tmp = nn.functional.relu(P_tmp - torch.mean(P_tmp.topk(2,-1)[0],-1).unsqueeze(-1))
            # normalize
            P_tmp = torch.divide(P_tmp, P_tmp.topk(1,-1)[0])
            # sparsify
            P_tmp = P_tmp.to_sparse()
            # concatenate
            if P_hat is None: P_hat = P_tmp
            else: P_hat = torch.cat([P_hat,P_tmp], dim=-2)

        return P_hat

# geometry generalization layer
class GeGeLayer(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.channels = in_shape[0]
        self.in_size = np.prod(in_shape[1:])
        self.out_shape = list(out_shape)
        self.out_size = np.prod(self.out_shape)
        assert self.out_size>=self.in_size, "out_size {} ({}) less than in_size {} ({})".format(self.out_size, self.out_shape, self.in_size, self.in_shape)
        self.padding = self.out_size - self.in_size
        self.sort = HardSort() # todo: configurable hyperparams?
        self.hidden = None

    def set_hidden(self, hidden):
        self.hidden = hidden

    def forward(self, x):
        # flatten all but first 2 dims (batch, channels)
        x = torch.reshape(x, (x.size()[0], x.size()[1], self.in_size))
        # add fake cells at the end to match output size
        x = nn.functional.pad(x, pad = (0, self.padding))
        # apply hidden layers to get score vector
        score = self.hidden(x)
        # "sort" inputs by score: produces NxN matrix of probabilities (1 or 0 w/ hard sort) for each input cell to move to each output cell
        probs = self.sort(score.squeeze())
        # "apply" the sort: sum up input cells in proportion to their probability (1 or 0 w/ hard sort) to be in a given output cell
        x = torch.bmm(probs, x.transpose(1,2)).transpose(1,2)
        # reshape to new geometry
        x = torch.reshape(x, [x.size()[0], x.size()[1]] + self.out_shape)
        # "reverse" the "sort" by inverting the probability matrix
        # hard sort: matrix is orthogonal, so inverse = transpose
        probs = torch.transpose(probs, probs.dim()-1, probs.dim()-2)
        return x, probs

    # starting from new geometry, restore original order, size, shape
    def restore(self, x, probs):
        restore_shape = [x.size()[0], x.size()[1]] + self.in_shape[1:]
        x = torch.reshape(x, (x.size()[0], x.size()[1], self.out_size))
        x = torch.narrow(x, x.dim()-1, 0, self.in_size)
        x = torch.bmm(probs, x.transpose(1,2)).transpose(1,2)
        x = torch.reshape(x, restore_shape)
        return x

# wrap existing NN in GeGe layer
class GeGeWrapper(nn.Module):
    def __init__(self, geom_layer, model):
        super().__init__()
        self.geom_layer = geom_layer
        self.model = model

    def forward(self, x):
        x, probs = self.geom_layer(x)
        x = self.model(x)
        x = self.geom_layer.restore(x, probs)
        del probs
        return x

def make_GeGeModel(model, in_shape, out_shape, hidden_layer_sizes, hidden_act):
    gege_layer = GeGeLayer(in_shape, out_shape)
    gege_hidden = [
        nn.Linear(gege_layer.out_size, hidden_layer_sizes[0]),
        hidden_act()
    ]
    for counter in range(len(hidden_layer_sizes)-1):
        gege_hidden.extend([
            nn.Linear(hidden_layer_sizes[counter], hidden_layer_sizes[counter+1]),
            hidden_act()
        ])
    gege_hidden.append(nn.Linear(hidden_layer_sizes[-1], gege_layer.out_size))
    hidden_model = nn.Sequential(*gege_hidden)
    gege_layer.set_hidden(hidden_model)
    return GeGeWrapper(gege_layer, model)
