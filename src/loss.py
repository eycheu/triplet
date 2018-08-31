import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineTripletLoss(nn.Module):
    """Online Triplets Loss
    Take a batch of embeddings and corresponding labels.
    Triplets are generated using TripletSelector object that takes embeddings and targets and returns the indices of
    triplets.
    
    Arguments:
    margin: (float): Additional marginal distance between anchor and negative points.
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

    
class OnlineNonLinearTripletLoss(nn.Module):
    """
    #https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
    
    Arguments:
    embed_dim: (int): The number of embedding dimensions.
    beta1: (float): The scaling factor, embed_dim is recommended.
    beta2: (float): The scaling factor for anchor-negative loss term, slightly more than embed_dim is recommended.
    epsilon: (float): The Epsilon value to prevent ln(0).
    """
    def __init__(self, triplet_selector, max_dist=2, beta1=2, beta2=2.01, epsilon=1e-8):
        super(OnlineNonLinearTripletLoss, self).__init__()
        self.triplet_selector = triplet_selector
        self.max_dist = max_dist
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).norm(p=2, dim=1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).norm(p=2, dim=1)
        
        pos_dist = -torch.log(-torch.div(ap_distances, self.beta1) + 1 + self.epsilon)
        neg_dist = -torch.log(-torch.div(self.max_dist - an_distances, self.beta2) + 1 + self.epsilon)
        
        loss = neg_dist + pos_dist
        
        return loss.mean(), len(triplets)
