# Lab 1

## What is the impact of varying batch sizes and why?
#### Training time and computational efficiency
Larger batch sizes means that more computations will take place in parallel. This implies a more efficient usage of the hardware since GPUs are designed for high data parallelism. Hence larger batch sizes will decrease the training time.

#### Generalisation
Larger batch sizes can lead to poor generalisation because the model will overfit to the training data. This can happen because larger batch sizes will provide a more accurate gradient and hence will be more robust to noise.

Smaller batch sizes can lead to better generalisation because the gradients will be more sensitive to noise in the data therefore less likely to overfit to training data.

#### Memory usage
Larger batch sizes require more memory in between layers due to backpropagation.

## What is the impact of varying maximum epoch number?
#### Underfiting vs Overfitting
If the max epoch number is too small then it might lead to underfitting because the model was not given enough time to learn the underlying patterns in the data.

If the max epoch number is too big then it might lead to overfitting because the model was given too much tim to learn the underlying patterns in the dataincluding noise and outliers.

Solution: Early stopping - stop training if validation loss stagnates or worsens.

#### Training time
The training time has positive correlation with the max epoch number.

## What is happening with a large learning rate and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?
#### Large learning rates
Convergence happens may happen faster therefore smaller training time.
The model may overshoot the global optimum which might cause the optimisation processes to osciallate or diverge.

#### Small learning rates
Convergence happens slower therefore larger training time.
The model is less likely to overshoot the global optimum therefore optimisation process is less likely to oscillate or diverge and eventually reach a global optimum.

#### Relationship between learning rates and batch sizes
When the batch size is very larget the gradient calculations are more accurate since there is more data to estimate them hence less suceptible to noise in the data set. Therefore a larger learning rate might be prefered to 

#TODO: do I need to collect data to support my claims?
