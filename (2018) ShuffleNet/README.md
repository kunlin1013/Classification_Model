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

### Channel shuffle process
![Channel shuffle process](https://github.com/kunlin1013/Classification_Model/blob/main/(2018)%20ShuffleNet/img/Channel%20shuffle%20process.png)

### Architecture
![Block](https://github.com/kunlin1013/Classification_Model/blob/main/(2018)%20ShuffleNet/img/Block.png)

(a) the basic ShuffleNet unit

(b) the ShuffleNet unit for spatial down sampling (2×)

(c) the ShuffleNetV2 basic unit

(d) the ShuffleNetV2 unit for spatial down sampling (2×)

![architecture](https://github.com/kunlin1013/Classification_Model/blob/main/(2018)%20ShuffleNet/img/architecture.png)

### Novelty
- 4 guidelines
  - **Equal channel** width minimizes memory access cost (MAC)
  - Excessive group convolution increases MAC
  - Network fragmentation reduces degree of parallelism.
    - e.g. In the **Inception model**, the faster branches must wait for the slower branches among the four branches.
  - Element-wise operations are non-negligible.
- channel split
