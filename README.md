# Code for "Out-of-Distribution Knowledge Distillation via Confidence Amendment"

##Standard Network
Create a folder named "checkpoints," and download the standard network from the link below. Place it inside this folder.
* [The standard network learned from CIFAR10](https://drive.google.com/file/d/1k3f2XopwrreyXG7M4mW5ANZX317JZK6Z/view?usp=sharing)
* [The standard network learned from Imagenet](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#ResNet50_Weights)

## Datasets
### In-distribution Datasets
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [Imagenet](https://www.image-net.org/)

### Out-of-Distribtion Datasets
Create a folder named "data," and download the OOD datasets from the link below. Place it inside this folder.
* [CUB200](https://www.vision.caltech.edu/datasets/cub_200_2011)
* [StanfordDogs120](http://vision.stanford.edu/aditya86/ImageNetDogs)
* [OxfordPets37](https://www.robots.ox.ac.uk/~vgg/data/pets)
* [Oxfordflowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers)
* [Caltech256](https://www.kaggle.com/jessicali9530/caltech256)
* [DTD47](https://www.robots.ox.ac.uk/~vgg/data/dtd)
* [COCO](https://cocodataset.org)

## Run CA-
```python
python Generator.py --T 1000 --gap 200 --lr 0.05 --save --generator 'DI'
python Discriminator.py --epoch 200 --T 1000 --gap 200 --generator 'DI' --alpha 0.1
```

## Run CA+
```python
python Generator.py --T 1000 --gap 200 --lr 0.05 --save --generator 'DR'
python Discriminator.py --epoch 200 --T 1000 --gap 200 --generator 'DR' --alpha 0.1
```

