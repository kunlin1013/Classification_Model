# ShuffleNetV2 (2018)
### Basic information
- paper name: [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)
- author: Ningning Ma
- from: Megvii Inc.
- ECCV 2018
- Group convolution can reduce parameters and computational load, but there is **no exchange of information between different groups within the group convolution**.
- ShuffleNetV1 introduced channel shuffle to facilitate the exchange of information across different channels.
- ShuffleNetV2 proposes that computational complexity cannot be solely assessed by FLOPs.
- It introduces 4 methods for designing efficient networks and presents a new block design.

### Architecture

### Novelty
- 4 guidelines