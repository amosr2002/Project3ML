# Project3ML


## Description of Experiments
The experiments we ran consisted of adding more layers, more Units, experimenting with different activation/optimization functions, using a minmax scaler (between 0 and 1) and also trying out convolutional/pooling layers to see if we could achieve better accuracy scores. For example, we tried adding two hidden layers with 64 units using Adadelta and Selu but the test score was only 0.1977, so we then tried adding more layers and changing the optimizer. We also tried to used GridSearchCrossValidation to see if we could find the best optimizer in terms of the cross validated score, but it was too time expensive to run completely. 

ANN Architecture Experiments:
1. We tried a convolutional network to try to see if we can decompress some of the information and also a Pooling Layer to further reduce the dimensions. The number of neurons in the Convolutional layer was 84 and the kernel size was 3 to 3 and the number of units in the dense layer was 84 units. The Relu activation function was used and the SGD optimizer was used with only 11 epochs because it took a very long time to train. We used a minmax scaler with a feature range of (0,1) because it achieved us better results than the default scaler provided.

Test Accuracy = 0.27
 ```
{
 model.add(layers.Conv2D(84, (3, 3)))
model.add(Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dense(84))
model.add(Activation('relu'))
model.add(layers.Dropout(0.5))
}
```

2. 

We also tried usic a simpler network with only only one convolutional and pooling layer, but we tried to experiment with the SeparableConv2D because it is suppose to support faster computations. We used 64 units for the SeparableConv Layer and 64 Units for the Dense layer and a Selu Activation Function. We trained using 100 Epochs and the Adam Optimizer, but it took way too long to train and exceeded the maximum run time. The minmax scaler with a feature range of (0,1) was also used for this architecture
 ```
{
## Test Accuracy 0.41, but took way too long to Train, Used 100 epochs
# model.add(layers.SeparableConv2D(64, (3, 3), input_shape=(32,32,1)))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='selu'))
}
```

3. 

We tried adding two hidden layers with 64 units and using a Selu Activation Function. We also trained it on 80 epochs and used the Adadelta Optimizer and recieved a test accuracy of 0.1977.  The minmax scaler with a feature range of (0,1) was also used for this architecture
 ```
{
## Test Accuracy 0.1977, Used Adadelta Activation Function
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='selu'))
# model.add(layers.Dense(64, activation='selu'))

}
```

4. 

We tried adding 4 Hidden Layers with a Relu activation function and a SGD Optimizer. This was trained using 100 Epochs and 64 units were used in each of the hidden layers. The minmax scaler with a feature range of (0,1) was also used for this architecture
 ```
{
##Test Accuracy 0.33, Used SGD Optimizer
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
}
```

5. 

We tried adding 4 Hidden Layers with a mixture of selu, tanh, and relu activation functions and an Adam Optimizer. This was trained using 100 Epochs and 63, 84, 90, and 110 units were used in each of the hidden layers respectively. The minmax scaler with a feature range of (0,1) was also used for this architecture.
 ```
{
##Test Accuracy 0.39, Used Adam Optimizer
# model.add(layers.Flatten())
# model.add(layers.Dense(63, activation='selu'))
# model.add(layers.Dense(84, activation='selu'))
# model.add(layers.Dense(90, activation='tanh'))
# model.add(layers.Dense(110, activation='relu'))
}
```

6. 

We tried adding 4 Hidden Layers with a Relu activation function and an Adam Optimizer. The Adam optimizer was working well so we stuck with it. This was trained using 100 Epochs and 64 units units were used in each of the hidden layers respectively. The minmax scaler with a feature range of (0,1) was also used for this architecture.
 ```
{
#Test Accuracy 0.38, Used Adam Optimizer
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
}
```

7. 

We tried adding 4 Hidden Layers with a selu activation function and an Adam Optimizer. The Adam optimizer was working well so we stuck with it. This was trained using 100 Epochs and 94 units units were used in each of the hidden layers respectively. The minmax scaler with a feature range of (0,1) was also used for this architecture. We achieved a test accuracy of 0.401.
 ```
{
##Test Accuracy 0.401, Used Adam Optimizer, MinMax Achieved 0.01 better Accuracy than the Default Scaler
# model.add(layers.Flatten())
# model.add(layers.Dense(94, activation='selu'))
# model.add(layers.Dense(94, activation='selu'))
# model.add(layers.Dense(94, activation='selu'))
# model.add(layers.Dense(94, activation='selu'))
}
```

8. 

We tried adding 4 Hidden Layers with a selu activation function and an Adam Optimizer. The Adam optimizer was working well so we stuck with it. This was trained using initially 100 Epochs, but after looking at the train/validation curve we decreased it to 80 epochs. We also tried decreasing the hidden unit size to 64 units for each layer. The minmax scaler with a feature range of (0,1) was also used for this architecture. We achieved a test accuracy of 0.405.
 ```
{
#Test Accuracy 0.405, Used Adam Optimizer, MinMax Achieved 0.01 better Accuracy than the Default Scaler
# Weighted Precision Score:  0.4034959726568194
# WeightedRecall Score:  0.4048
#parser.add_argument("--hidden_size", default=64, type=int)
model.add(layers.Flatten())
model.add(layers.Dense(args.hidden_size, activation='selu'))
model.add(layers.Dense(args.hidden_size, activation='selu'))
model.add(layers.Dense(args.hidden_size, activation='selu'))
model.add(layers.Dense(args.hidden_size, activation='selu'))
}
```

## Description of the Best Model/Training Procedure

After concluding the experiments, we decided that our best model was a network which included four hidden layers with 64 units in each of them. We also saw better performance in the Selu activation function and the Adam optimizer, so we ended up choosing those. We also initially trained the model using 100 epochs but decreased it to 80 epochs after examining the train/validation curve. The default batch size of 512 was used and the Adam schocashtic gradient descent optimizer was used. A minmax scaler using a feature range of (0,1) was used because we observed better performance than using the default scaler provided.

 ```
{
#Test Accuracy 0.405, Used Adam Optimizer, MinMax Achieved 0.01 better Accuracy than the Default Scaler
# Weighted Precision Score:  0.4034959726568194
# WeightedRecall Score:  0.4048
#parser.add_argument("--hidden_size", default=64, type=int)
model.add(layers.Flatten())
model.add(layers.Dense(args.hidden_size, activation='selu'))
model.add(layers.Dense(args.hidden_size, activation='selu'))
model.add(layers.Dense(args.hidden_size, activation='selu'))
model.add(layers.Dense(args.hidden_size, activation='selu'))
}
```

## Training Performance Plot
![best_train_val](https://user-images.githubusercontent.com/77814810/195717557-d40d8cb9-198e-400f-badd-52591274979b.png)

## Performance of Best Model
Test Accuracy:  0.4047999978065491

Weighted Precision and Recall Scores
Precision Score:  0.4034959726568194
Recall Score:  0.4048

## Confusion Matrix of Best Model
![best_confusion](https://user-images.githubusercontent.com/77814810/195717639-d36541a8-ef8b-4ce7-808b-16a02425c2f1.png)

## Visualization of Misclassified Images
![cat_misclassified](https://user-images.githubusercontent.com/77814810/195718839-33dd6e3a-35d9-421d-a2c7-c32139b23112.png)
**Cat was Misclassified as a Ship**

![airplane_misclassified](https://user-images.githubusercontent.com/77814810/195719791-fa64bedf-cd96-4061-81bb-b517b140b543.png)
**Airplane was misclassified as a Ship**

![frog_misclassified](https://user-images.githubusercontent.com/77814810/195719909-4ea8a881-3c7c-4a37-831d-dfc67cbd3594.png)
**Frog was Misclassified as a Deer**





