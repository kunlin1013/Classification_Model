## AlexNet (2012) - the epoch-making ancestor of Deep Learning
- First demonstration that learned features are better than manually designed ones.

![AlexNet](https://github.com/kunlin1013/Classification_Model/blob/main/AlexNet%20(2012)/img/Architecture.png)

### Basic information
- paper name: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- author: Hinton (Teacher), Alex Krizhevsky (Student)
- from: University of Toronto
- ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 2012 Champion :trophy:
- The classification accuracy has been improved from the traditional 70%+ to 80%+.

### Novelty
1. First use of **GPU** for accelerated network training.
2. Use the **Relu** for activation function. Traditionally, sigmoid or tanh are used as activation functions, but these functions can cause the gradient to vanish as the model depth increases.
3. Use the **Local Response Normalization (LRN)**.
4. Use **Dropout** in the first two layers of the fully connected layers to prevent overfitting.

### Suppress overfitting
1. Data Augmentation
   - After image translations and horizontal reflections, a 224x224 image is randomly cropped from the 256x256 image to serve as the image for training.
   - Use PCA to randomly alter the color channels.
2. Dropout
   - This will cause the training speed to slow down.

### [Local Response Normalization (LRN)](<https://towardsdatascience.com/difference-between-local-response-normalization-and-batch-normalization-272308c034ac>)
**AlexNet employs inter-channel Local Response Normalization (LRN).**
![LRN1](https://github.com/kunlin1013/Classification_Model/blob/main/AlexNet%20(2012)/img/LRN_1.jpeg)

$$
b^i_{x,y} = a^i_{x,y} / \left(k + \alpha \sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)} (a^j_{x,y})^2 \right)^\beta
$$
   
In AlexNet, each filter output $a^i_{x,y}$ at the pixel location $(x,y)$ is normalized by considering the activity of neighboring filters at the same position. This operation results in the normalized output $b^i_{x,y}$ .  The total number of channels in the layer is denoted by $N$, and the normalization process involves a set of hyper-parameters $(k,\alpha,\beta,n)$. Here $k$ prevents division by zero, $\alpha$ scales the normalization, and $\beta$ provides contrast between the outputs. The hyper-parameter $n$ defines the size of the neighborhood, which is how many adjacent filter responses are considered for the normalization process. In AlexNet, the standard choice of these parameters is $k=2, \alpha=10^-4, \beta=0.75, n=5$.

However, in the example illustrated below, we set the hyper-parameters to $(k,\alpha,\beta,n)=(0,1,1,2)$, with $n=2$ indicating that when normalizing an element, we only consider the element's immediate predecessor and successor along the channel dimension (i.e., in a one-dimensional context). In cases where boundaries are encountered, padding with zero is used to complete the set. This means that when calculating the normalized value for the point $(i, x, y)$, it is sufficient to only consider the values at $(i-1, x, y)$, $(i, x, y)$, and $(i+1, x, y)$, with any values beyond the boundaries assumed to be zero.

![LRN2](https://github.com/kunlin1013/Classification_Model/blob/main/AlexNet%20(2012)/img/LRN_2.jpeg)

In the example above, different colors represent different channels. Taking the point in the upper left corner (coordinate $(0,0,0)$ ) of the first feature map as an example, its value before normalization is $1$, and after normalization, it becomes $0.5$. In calculating this value, we must consider the values before and after it. Since there is no value for $(i-1, x, y)$ — that is a boundary condition — we default it to $0$, and $(i+1, x, y)$ is the value in the upper left corner of the orange feature map, which is $1$. Therefore, substituting into the formula, we obtain the normalized value as: $b=1/(0+1*(0^2+1^2+1^2)^1)=0.5$. Thus, the normalized value for the point at coordinate (0,0,0) is calculated. This process is repeated similarly for other points.

The calculation for the coordinate point $(1,0,0)$ is as follows: $b=1/(0+1*(1^2+1^2+2^2)^1)=0.17$.

