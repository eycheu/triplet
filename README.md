# Learning Constellation using Triplets

The idea is to learn a Deep Neural Network using triplets. Triples consist of an anchor point, a positive point, and a negative point. Anchor and positive points belong to the same classification, category, or class label while negative point that is not of the same class label as anchor point is a point close to the anchor point in the embedding space. The objective is to learn a complex function such that anchor points eventually "move" close to all other points of the same class label and far away to all other negative points in the embedding space.

Illustration of the learning algorithm using PyTorch in producing the embedding space is shown below.

![Demo1](https://github.com/eycheu/triplet/blob/master/image/demo1.png)

This simple demonstration utilised a L2 Norm clamped embedding space instead of a L2 Norm space in [1], and the following non-linear triplet loss function is adapted from [2].

![Triplet loss](https://github.com/eycheu/triplet/blob/master/image/nonlinear_triplet_loss.png)

The non-linear triplet loss consists of two terms. The first term is an associative term to move anchor closer to positive point. The second term is a contrastive term to move anchor further away from the negative point. For the example, the maximum distance D between two points in the embedding space is 2, and other parameters are chosen such that the loss terms are as shown below. Loss is 0 when distance between anchor point and positive point is 0 and when anchor point is farthest from the negative point.

![Triplet loss terms](https://github.com/eycheu/triplet/blob/master/image/nonlinear_triplet_loss_terms.png)

The learning methodology is extended to the next example with dummy data randomly sampled into 10 classes. This experiment exemplifies real world examples corrupted with noise. The same learning triplet loss function is used. The final training loss is not 0 since the negative points are closer to the anchor (or positive) points; all the points now have to share a relatively smaller embedding space than the first example. Outliers or mislabelled data points could be identified using this encoding.

![Demo2](https://github.com/eycheu/triplet/blob/master/image/demo2.png)

While the examples show the use of embedding space to verify data patterns, recognise differences, and clusters similar points, this learning method can also be applied to channel equalisation. The positive point can be a fixed point in the embedding space that represents a symbol in a constellation diagram. An embedding network could be periodically tuned to learn a reversal of distortion incurred when signal is transmitted through a channel.

## References

[1] Florian Schroff, Dmitry Kalenichenko, James Philbin. [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832). CVPR, 2015.

[2] Marc-Olivier Arsenault, A more efficient loss function for Siamese NN, https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24.
