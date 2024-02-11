# Advanced Deep Learning Systems: Lab 4 (Software)

**Basheq Tarifi**

![bkt123](https://img.shields.io/badge/short%20code-bkt123-green) ![02482739](https://img.shields.io/badge/CID-02482739-blue) ![AML](https://img.shields.io/badge/Course-MSc%20Applied%20Machine%20Learning-purple)

## Overview
This lab provided an investigation into using the search in MASE to implement a network architecture search. A network architecture search can be done in many ways, such as changing layer dimensions or properties or even adding layers. This lab involves scaling layers in a linear network to find the best performing network.

## Channel Multiplier Pass
A basic channel multiplier was initially done on the example `JSC_Three_Linear_Layers` network. 

The transform works as follows:
```python
for layer in layers:
    if layer matches config:
        transform layer by modifying the size
```

The provided pass accepts the following configuration:
```python
pass_config = {
    "by": "name",
    "default": {"config": {"name": None}},
    "seq_blocks_2": {
        "config": {
            "name": "output_only",
            "channel_multiplier": 2,
        }
    },
    "seq_blocks_4": {
        "config": {
            "name": "both",
            "channel_multiplier": 2,
        }
    },
    "seq_blocks_6": {
        "config": {
            "name": "input_only",
            "channel_multiplier": 2,
        }
    },
}
```

In implementing this as a pass, there is a problem posed by this transform in that:
- the layers, such as the `relu` layers in between the `linear` layers, also need to be transformed
- there may be mismatches in the layers
- there should be a possibility of scaling the input channel and output channel asymmetrically

To solve the first problem, the pass was written such that for every node being transformed, the previous (and subsequent) layers are checked. Layers such as `relu` and `batchnorm1d` cannot have a different input and output, so in order to successfully transform any layer there needs to be at least one linear layer before or after it. Naturally, this means that the input of the first layer and the output of the last layer cannot be scaled. All previous layers (up to and including the output of the previous `linear` layer), and/or all subsequent layers (up to and including the input of the next `linear` layer) are transformed to match the current channel being scaled. This leads to the second issue of mismatched scaling. To solve this, any subsequent scalings overwrite previous scalings. Finally, asymmetric scaling can be achieved by adding parameters for input and output scaling:
```python
pass_config = {
    "by": "name",
    "default": {"config": {"name": None}},
    "seq_blocks_2": {
        "config": {
            "name": "output_only",
            "channel_multiplier": 2,
        }
    },
    "seq_blocks_4": {
        "config": {
            "name": "both",
            "input_multiplier": 2,
            "output_multiplier": 4,
        }
    },
    "seq_blocks_6": {
        "config": {
            "name": "input_only",
            "channel_multiplier": 4,
        }
    },
}
```

The final pass is called `linear_multiplier_transform_pass` and is implemented in [linear_multiplier.py](../../../machop/chop/passes/graph/transforms/layers/linear_multiplier.py) ([remote link](https://github.com/btarifi10/mase/tree/btarifi/dev/machop/chop/passes/graph/transforms/layers/linear_multiplier.py)).

This allows us to obtain the desired network with asymmetric scaling. Performing the `report_graph_analysis_pass` produces the following on the `JSC_Three_Linear_Layers` network, which matches the desired network:
```
Layer types: [
    BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    ReLU(inplace=True),
    Linear(in_features=16, out_features=32, bias=True),
    ReLU(inplace=True),
    Linear(in_features=32, out_features=64, bias=True),
    ReLU(inplace=True),
    Linear(in_features=64, out_features=5, bias=True), ReLU(inplace=True)
]
```

## Channel Multiplier Search
The above pass was then incorporated into a search space to be used with any of the search strategies. This will allow a search to evaluate different network architectures with different layer sizes to find the best performing network.

The search space, called `LinearChannelMultiplierSpace` (with key `transform/linear_channel_multiplier`) inherits and implements the `SearchSpaceBase` and is implemented in [linear.py](../../../machop/chop/actions/search/search_space/transformation/linear.py) ([remote link](https://github.com/btarifi10/mase/tree/btarifi/dev/machop/chop/actions/search/search_space/transformation/linear.py)). It provides the total space of possible configurations based on the inputs in the `toml` file.

The search was performed using the Optuna TPE strategy with the new `transform/linear_channel_multiplier` space. In order to actually evaluate the different networks, the search strategy needs to train the network during the search. This can be done using the [`basic_train` runner](../../../machop/chop/actions/search/strategies/runners/software/train.py) ([remote link](https://github.com/btarifi10/mase/tree/btarifi/dev/machop/chop/actions/search/strategies/runners/software/train.py)) runner. _However_, this runner was not functional.