# Requirements
 torch >= 1.1.0

 numpy >= 1.16.2

 tqdm >= 4.31.1

 visdom >= 0.1.8.8
 
 torchvision == 0.2.1
 
 download STL10 dataset b4 using it.

# To Activate Visdom Server
  visdom -port 9000

# Parameters

* --data (default CIFAR10): choose dataset out of CIFAR10, CIFAR100, MNIST, FashionMNIST, SVHN, STL10, TINY(TinyImageNet), IMAGENET(Note that TINY and IMAGENET should be downloaded by yourself at directories '.' and '/nas/Public/', respectively.
* --model (default resnet18): Backbone architecture.(resnet18, resnet34)
* --rule (default Random) : picking rule/strategy.(Random, PredictedLoss, Entorpy, Margin, lrl)
* --lamb1 : Hyperparameter for MarginRankingLoss
* --lamb2 : Hyperparameter for LogRatioLoss
* --lamb3 : Hyperparameter for TripletLoss
* --lamb4 : Hyperparameter for LossTripletLoss
* --lamb5 : Hyperparameter for Emb dim=10 to Emb dim=10
* --lamb6 : Hyperparameter for L2 weight regularization
* --lamb7 : Hyperparameter for GAN ActiveLearning
* --lamb8 : Hyperparameter for DCGAN ActiveLearning
* --lamb9 : Hyperparameter for CustomTripletLoss
* --lambt : Hyperparameter for TripletLoss with Loss
* All above lamb's are default 0.
* --everyinit : Fix initial weights at every cycle.
* --cycle (default 10) : The number of cycles
* --trials (default 4) : The number of trials
* --query (default 1000) : The number of queries at every cycle.
* --softmax (default False) : use (10-dim) softmax outputs of backbone model instead of 512-dim feature embeddings.

 
# Example Codes

* Learning Loss for Active Learning.
```
python3 main.py --lamb1=1 --rule PredictedLoss --data CIFAR10 --query=1000 --trials=4
```

* Learning Loss for Active Learning with initialization at every cycle.
```
python3 main.py --lamb1=1 --rule PredictedLoss --everyinit
```

* Regularization : TripletLoss with Loss (When you use this Loss, batch size and query should be multiple of 3.)
* Strategy : Random
```
python3 main.py --lambt=1 --rule Random --query=999 --batch=126
```
* Strategy : Entropy
```
python3 main.py --lambt=1 --rule Entorpy --query=999 --batch=126
```

# Additional Codes
* Regularization : LogRatioLoss
* Strategy : LogRatioLoss pick
* Dataset : CIFAR100
```
python3 main.py --lamb2=1 --rule lrl --data CIFAR100
```
* Regularization : Triplet Loss on Backbone Embeddings
* Strategy : Margin
* Dataset : FashionMNIST
```
python3 main.py --lamb3=1 --triplet --rule Margin --data FashionMNIST
```

* Regularization : TripletLog Loss on Loss Embeddings
* Strategy : LogRatioLoss pick
* Dataset : STL10
```
python3 main.py --lamb4=1 --Ltripletlog --rule lrl --data STL10
```
* Regularization : LiftedStructured Loss on Loss Embeddings
* Strategy : LogRatioLoss pick
* Dataset : SVHN
* Fix seed
```
python3 main.py --lamb4=1 --Lls --rule lrl --data SVHN --seed
```


# Reference

 Original reproduction of Learning Loss for Active Learning [Yoo et al. 2019 CVPR] : https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
 Its reproduced results
 ![Results](./results.PNG)
