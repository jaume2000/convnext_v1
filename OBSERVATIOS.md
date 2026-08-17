# Learning Notes

This is weekend project I made to learn how modern neural networks are trained on relatively big datasets such as ImageNet and go down with the coding. I tried to not use AI in the majority of the tasks except for debugging weights or optimizing the pre-allocation in RAM of the Leonardo supercomputer, where I trained the model.

Here I'll let the things that I learned.

* Always use the max workers possible: Read how many cpu threads do you have and use them to augment the data.
* The bottleneck never is the data augmentation, is the data reading: 32 ms for bringing from disk to RAM and then to GPU. 1 ms to augment the whole batch.
* If you can, allocate all your dataset in RAM, here, we moved 70 GB to the leonardo's RAM (compressed Imagenet dataset). That speeded up the training x8.
* AMP is garbage (for this case). From the first moment I used AMP with bfp16, the model started learning, and when reaching ~10% acc on train, it started droping back to 0.1% (random) with loss of 6.9 = ln(1000), that means that the model made random guess, this happend becasue the softmax was saturated -> Weights exploded, specially on LayerScale weights.
* Filter the Weight decay: Do not use weight decay in LayerNorm or LayerScale. These params showld be L2-free!
* Ensure to print your gradient norm, it should be arround O(1), if it starts going to 10, 50 or 100, or, 0.1 0.01, is that your model has a problem regarding the normalizing layers.

The biggest problem I encountered was the AMP on normalizing layers, for some reason, the weights of layer scale keept growing and growing around epoch 7-10 and the network just started to unlearn, even when reached 10% accuracy it got back to 0.1 (that is, random choice)