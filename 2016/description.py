"""
Process train data:
To synthesize the low-resolution
samples {Yi}, we blur a sub-image by a Gaussian kernel,
sub-sample it by the upscaling factor, and upscale it by
the same factor via bicubic interpolation:
 1. Cut big image into f_sub = 33 size squares
 2. sub-sample by upscaling factor
 3. apply blur

The upscaling factor is 3.
Try using the Set14 image dataset.

Init:
The filter weights of each layer are initialized by drawing
randomly from a Gaussian distribution with zero mean
and standard deviation 0.001 (and 0 for biases). 

Learning:
The learning rate is 10e-4 for the first two layers, 
and 10e-5 for the last layer. 
We empirically find that a smaller learning rate in the last layer is important 
for the network to converge.
"""