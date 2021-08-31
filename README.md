# lottery-ticket-distributions
CMSC828W Project - Lottery Ticket Distributions

### Abstract
Previous works have hypothesized and empirically shown the existence of so-called “lottery ticket” subnetworks within deep neural networks (Frankle & Carbin, 2018). These subnetworks have less than 20% the parameter size of the original model yet perform better with regards to training convergence, inference time, and test accuracy. The existence of such networks begs further exploration on their unintuitively high performance. In this paper, we look at two specific parameters of the lottery ticket: the distribution of non-zero weights after a lottery ticket has been found and the specific mask of the lottery ticket. With regards to these parameters, we test four separate network architectures: the original full network, the pruned lottery ticket network, the lottery ticket reinitialized with the empirical lottery ticket weight distribution, and a network with parameter size equal to the lottery ticket initialized with the empirical lottery ticket weight distribution. These architectures were experimentally tested on a fully-connected network and convolutional network for MNIST and CIFAR10. The test accuracy of these new architectures were worse than both the full network and the lottery ticket network, and the number of epochs to train to that accuracy increased significantly. At a high level, this shows that the distribution of weights and topology of the pruning mask do not provide adequate information to create a performance-equivalent subnetwork to the original network. Our findings imply that generating pruned networks highly depends on the initialization of the original network, both with regards to the exact pruning topology as well as the placement of model weights.



Final paper located in `lottery_ticket_initializations.pdf`

Written by Kusal De Alwis, Dhruv Maniktala, and Bryce Toole