from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from time import time
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import confusion_matrix
from tensorflow.keras import datasets, layers, models
import seaborn as sns


# PLS DO NOT EXCEED THIS TIME LIMIT
MAXIMIZED_RUNNINGTIME=1000
# REPRODUCE THE EXP
seed = 123
random.seed(seed)
np.random.seed(seed)
keras.utils.set_random_seed(seed)

parser = ArgumentParser()
###########################MAGIC HAPPENS HERE##########################
# Different hyper-parameters will greatly influence the performance.
# Hint: Advanced optimizer will achieve better performance.
# Hint: Large Epochs will achieve better performance.
# Hint: Large Hidden Size will achieve better performance.
parser.add_argument("--optimizer", default='sgd', type=str)
parser.add_argument("--epochs", default=75, type=int)
parser.add_argument("--hidden_size", default=32, type=int)
parser.add_argument("--scale_factor", default=1, type=float)
###########################MAGIC ENDS HERE##########################

parser.add_argument("--is_pic_vis", action="store_true")
parser.add_argument("--model_output_path", type=str, default="./output")
parser.add_argument("--model_nick_name", type=str, default=None)



args = parser.parse_args()
start_time = time()
# Hyper-parameter tuning
# Custom dataset preprocesst

# create the output dir if it not exists.
if os.path.exists(args.model_output_path) is False:
    os.mkdir(args.model_output_path)

if args.model_nick_name is None:
    setattr(args, "model_nick_name", f"OPT-{args.optimizer}-E-{args.epochs}-H-{args.hidden_size}-S-{args.scale_factor}")

'''
1. Load the dataset
Please do not change this code block
'''
class_names = {
    0: "airplane",
    1: "automobile",
    2:"bird",
    3:"cat",
    4:"deer",
    5:"dog",
    6:"frog",
    7:"horse",
    8:"ship",
    9:"truck"
}

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# check the validity of dataset
assert x_train.shape == (50000, 32, 32, 3)
assert y_train.shape == (50000, 1)

# Take the first channel
x_train = x_train[:, :, :, 0]
x_test = x_test[:, :, :, 0]

# split the training dataset into training and validation
# 70% training dataset and 30% validation dataset
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=seed, stratify=y_train)


if args.is_pic_vis:
    # Visualize the image
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i][0]])
    plt.show()


'''
2. Dataset Preprocess
'''
# Scale the image
###########################MAGIC HAPPENS HERE##########################
# Scale the data attributes 
# Hint: Scaling the data in the range 0-1 would achieve better results.

scaler = MinMaxScaler(feature_range=(0,args.scale_factor))

x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
x_valid = scaler.transform(x_valid.reshape(-1, x_valid.shape[-1])).reshape(x_valid.shape)
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)


###########################MAGIC ENDS HERE##########################

if args.is_pic_vis:
    plt.figure()
    plt.imshow(x_train[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()


'''
3. Build up Model 
'''
num_labels = 10
model = Sequential()
###########################MAGIC HAPPENS HERE##########################



# Build up a neural network to achieve better performance.
# Hint: Deeper networks (i.e., more hidden layers) and a different activation function may achieve better results.
model.add(Flatten())
model.add(Dense(args.hidden_size, activation="relu")) # first layer
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

###########################MAGIC ENDS HERE##########################
model.add(Dense(num_labels)) # last layer

# Compile Model
model.compile(optimizer=args.optimizer,

              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=args.epochs,
                    batch_size=512)
print(history.history)

# Report Results on the test datasets
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print("\nTest Accuracy: ", test_acc)


end_time = time()
assert end_time - start_time < MAXIMIZED_RUNNINGTIME, "YOU HAVE EXCEED THE TIME LIMIT, PLEASE CONSIDER USE SMALLER EPOCHS and SHAWLLOW LAYERS"
# save the model
model.save(args.model_output_path + "/" + args.model_nick_name)

'''
4. Visualization and Get Confusion Matrix from test dataset 
'''

y_test_predict = np.argmax(model.predict(x_test), axis=1)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

###########################MAGIC HAPPENS HERE##########################
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# param_grid = dict(optimizer=optimizer)
# keras_clf = KerasClassifier(model = model, epochs=100, batch_size=10, verbose=0)
# grid = GridSearchCV(estimator=keras_clf, param_grid=param_grid, n_jobs=-1, cv=5, scoring='accuracy')
# grid_result = grid.fit(x_train, y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# Visualize the confusion matrix by matplotlib and sklearn based on y_test_predict and y_test
# Report the precision and recall for 10 different classes
# Hint: check the precision and recall functions from sklearn package or you can implement these function by yourselves.

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

conf_matrix = confusion_matrix(y_test, y_test_predict)
sns.heatmap(conf_matrix)
plt.show()
###########################MAGIC ENDS HERE##########################






