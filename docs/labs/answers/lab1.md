# Lab 1

## What is the impact of varying batch sizes and why?

#### Results
**Batch size: 256** <br>
**Learning rate: 1e-5** <br>
**Max epochs: 5** <br>
Training time (seconds): 147
Test Accuracy: 0.46699467301368713
Test Loss: 1.4134204387664795

<<<<<<< HEAD
**Batch size: 128** <br>
**Learning rate: 1e-5** <br>
**Max epochs: 5** <br>
Training time: 276
Test Accuracy: 0.5100170969963074
Test Loss: 1.330588936805725

**Batch size: 64** <br>
**Learning rate: 1e-5** <br>
**Max epochs: 5** <br>
=======
**Batch size: 64** <br>
>>>>>>> Saving answers
Training time 522
Test Accuracy: 0.5219799280166626
Test Loss: 1.2776826620101929

<<<<<<< HEAD
=======
**Batch size: 16** <br>
Training time: 
Test Accuracy: 
Test Loss:
>>>>>>> Saving answers

#### Training time and computational efficiency
Larger batch sizes means that more computations will take place in parallel. This implies a more efficient usage of the hardware since GPUs are designed for high data parallelism. Hence larger batch sizes will decrease the training time.


#### Generalisation
Larger batch sizes can lead to poor generalisation because the model will overfit to the training data. This can happen because larger batch sizes will provide a more accurate gradient and hence will be more robust to noise.

Smaller batch sizes can lead to better generalisation because the gradients will be more sensitive to noise in the data therefore less likely to overfit to training data.


#### Memory usage
Larger batch sizes require more memory in between layers due to backpropagation needing the forward pass outputs of each layer.

## What is the impact of varying maximum epoch number?

#### Results
# TODO: collect this data.
# TODO: put this data into tables.
**Batch size: 256** <br>
**Learning rate: 1e-5** <br>
**Epoch number: 1** <br>

**Batch size: 256** <br>
**Learning rate: 1e-5** <br>
**Epoch number: 5** <br>

**Epoch number: 1** <br>
Training time: 42
Test Accuracy: 0.2872336208820343
Test Loss: 1.5318588018417358

**Epoch number: 5** <br>
Training time: 153
Test Accuracy: 0.2872336208820343
Test Loss: 1.5318588018417358

**Batch size: 256** <br>
**Learning rate: 1e-5** <br>
**Max epochs: 10** <br>
Training time: 286
Test Accuracy: 0.2872336208820343
Test Loss: 1.5318588018417358
#### Underfiting vs Overfitting
If the max epoch number is too small then it might lead to underfitting because the model was not given enough time to learn the underlying patterns in the data.

If the max epoch number is too big then it might lead to overfitting because the model was given too much tim to learn the underlying patterns in the dataincluding noise and outliers.

Solution: Early stopping - stop training if validation loss stagnates or worsens.

#### Training time
The training time has positive correlation with the max epoch number.

## What is happening with a large learning rate and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?

#### Results
**Batch size: 256** <br>
**Learning rate: 1e-5** <br>
**Max epochs: 5** <br>
Training time: 189
Test Accuracy: 0.46699467301368713
Test Loss: 1.4134204387664795

**Batch size: 256** <br>
**Learning rate: 1e-4** <br>
**Max epochs: 5** <br>
Training time: 183
Test Accuracy: 0.6248619556427002
Test Loss: 1.080784797668457

**Batch size: 256** <br>
**Learning rate: 1e-3** <br>
**Max epochs: 5** <br>
Training time: 186
Test Accuracy: 0.7136226892471313
Test Loss: 0.8588454723358154

#### Large learning rates
Convergence happens may happen faster therefore smaller training time.
The model may overshoot the global optimum which might cause the optimisation processes to osciallate or diverge.

#### Small learning rates
Convergence happens slower therefore larger training time.
The model is less likely to overshoot the global optimum therefore optimisation process is less likely to oscillate or diverge and eventually reach a global optimum.

#### Relationship between learning rates and batch sizes
When the batch size is very larget the gradient calculations are more accurate since there is more data to estimate them hence less suceptible to noise in the data set. Therefore a larger learning rate might be prefered to 

## Implement a network that has in total around 10x more parameters than the toy network.

```python
ass JSC_My(nn.Module):
    def __init__(self, info):
        super(JSC_My, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear              # 2
            # 2nd LogicNets Layer
            nn.BatchNorm1d(16),  # output_quant       # 3
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear              # 2
            # 3rd LogicNets Layer
            nn.BatchNorm1d(16),  # output_quant       # 3
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear              # 2

            # 4th LogicNets Layer
            nn.BatchNorm1d(16),  # output_quant       # 3
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear              # 2

            # 5th LogicNets Layer
            nn.BatchNorm1d(16),  # output_quant       # 3
            nn.ReLU(16),  # 1
            nn.Linear(16, 5),  # linear              # 2

            nn.ReLU(5),  # 4
        )

    def forward(self, x):
        return self.seq_blocks(x)
```

## Test your implementation and evaluate its performance.
JSC-Tiny
Test accuracy: 0.5126265287399292
Test loss: 1.326629400253296

JSC-My
Test accuracy: 0.6251558065414429
Test loss: 1.0560030937194824

The performance displayed by JSC-My is 21.8% better than for JSC-Tiny in terms of test accuracy and 20.3% better in terms of test loss.
The JSC-My network contains 1.3K parameters compared to 127 from JSC-Tiny.
#TODO: do I need to collect data to support my claims?
