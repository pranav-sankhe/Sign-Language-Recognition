# Final Presentation Notes 

## Slide 3


### Structure of Bi-LSTMs
LSTM in its core, preserves information from inputs that has already passed through it using the hidden state.
Unidirectional LSTM only preserves information of the past because the only inputs it has seen are from the past. Using bidirectional will run your inputs in two ways, one from past to future and one from future to past and what differs this approach from unidirectional is that in the LSTM that runs backwards you preserve information from the future and using the two hidden states combined you are able in any point in time to preserve information from both past and future.

### Pre-training RNN
With transfer learning, we can take a pretrained model, which was trained on a large readily available dataset (trained on a completely different task, with the same input but different output). We use the output of that layer as input features to train a much smaller network that requires a smaller number of parameters. This smaller network only needs to learn the relations for your specific problem having already learnt about patterns in the data from the pretrained model. This way a model trained to detect Cats can be reused to Reproduce the work of Van Gogh



### Problems with pre-training RNN
Pre-Training requires the input to be same. The pretraining dataset was skeletal data with 25 joints. Our data has 104 joints. The joints which are common between both the datasets were 12 which were very low. 
We trained the RNN network shown in figure on the pretraining datasets considering only the intersecting joints but the results were not impressive. We could achieve a bare accuracy of 15%. This explained why pre-training on RNNs is not very common as against convolutional neural networks. 
Hence we had to drop this model. 


## Slide 4

### Variance based feature selection 

### C3D data visualization 

### Skeletal Data Processing 

### Intersecting joints between our dataset and the pre-training dataset
