# Lab 3

## Task 1: Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.

* Latency of forward pass. - Completed
* Model size: memory requirements - Completed
* Number of FLOPs - TODO
* Confusion matrix metrics: precision, F1 scores and recall - Completed

## Task 2: Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).


## Task 3: Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.
```python
class SearchStrategyOptuna(SearchStrategyBase):
    is_iterative = False

    def _post_init_setup(self):
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        if not self.sum_scaled_metrics:
            self.directions = [
                self.config["metrics"][k]["direction"] for k in self.metric_names
            ]
        else:
            self.direction = self.config["setup"]["direction"]

    def sampler_map(self, name):
        match name.lower():
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "tpe":
                sampler = optuna.samplers.TPESampler()
            case "nsgaii":
                sampler = optuna.samplers.NSGAIISampler()
            case "nsgaiii":
                sampler = optuna.samplers.NSGAIIISampler()
            case "qmc":
                sampler = optuna.samplers.QMCSampler()
            # NOTE: added brute force sampler from Optuna into the SearchStrategyOptuna class,
            # then updated the toml configuration file to use "brute" sampler.
            case "brute":
                sampler = optuna.samplers.BruteForceSampler()
            case _:
                raise ValueError(f"Unknown sampler name: {name}")
        return sampler
```

## Task 4: Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.

### TPE sampler results
100% 20/20 [00:06<00:00,  3.22it/s, 6.20/20000 seconds]
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        4 | {'loss': 1.542, 'accuracy': 0.315} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.315, 'average_bitwidth': 0.4} |

### Brute force sampler results
90% 18/20 [00:05<00:00,  3.38it/s, 5.33/20000 seconds]
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |       14 | {'loss': 1.549, 'accuracy': 0.303} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.303, 'average_bitwidth': 0.4} |

### Comparison
* The number of samples used by the TPE sampler is 20 and by the Brute Force sampler is 18 therefore in this scenario the Brute force sampler used less samples.
* The accuracy achieved by the TPE sampler is 31.5% which is 1.2% more than the accuracy achieved by the Brute force sampler: 30.3%.
* The hardware metrics achieved by both samplers is the same as a result the found average bitwidth for both samplers is the same: 0.4.
* The number of iterations per second for the TPE sampler is 0.16 it/s smaller than for the Brute force sampler.
* The loss achieved by TPE is 0.007 units smaller than for Brute force.

### Conclusions
* The TPE sampler is expected to be more sample efficient than the Brute force sampler because it uses mathematical models to intelligently explore its search space while the Brute force sampler explores the search space exhaustively and without direction. However, it can be observed in the results that the Brute force sampler uses 2 less samples than the TPE sampler. From the documentation for the Brute Force sampler it can inferred that this sampler is prone to failing to search the entire search space thus justifying why it only uses 18 samples. Therefore the TPE sampler and Brute force sampler can be assumed to use the same number of samples.
* As aforementioned the efficiency of TPE is expected to be higher than the Brute force sampler, this is exemplied by the achieved accuracy of TPE being 1.2% higher and the achieved loss being 0.007 units smaller than the Brute force sampler.
* The TPE sampler is more computationally expensive than the Brute force sampler which is shown by the number of iterations per seconds achieved by the TPE sampler being 0.16 it/s smaller than for the Brute force sampler. 

