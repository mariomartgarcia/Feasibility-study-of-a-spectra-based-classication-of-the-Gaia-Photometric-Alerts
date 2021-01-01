#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import TensorBoard, History 
from sklearn.utils import class_weight
import timeit
import warnings
import seaborn as sns
import matplotlib.pyplot as plt 


#--------------------------------------------------------------------------------------------------------------------------
#SUBSET EXTRACTION
#--------------------------------------------------------------------------------------------------------------------------
#Function that extracts a balanced subset of a dimension from the initial dataset
def subset(dim, df):
    tic = timeit.default_timer()
    subdf=pd.DataFrame()
    a=[]
    for i in range(dim):
        while True:
            extract = df.sample(n=1)
            if extract["feature"].item()  not in a:
                break
        subdf = pd.concat([subdf, extract])

        value, num = np.unique( subdf["feature"], return_counts= True)  
        for j in range(len(num)):
            if (num[j]>=dim/8) & (value[j] not in a):
                a.append(value[j])

    toc = timeit.default_timer()
    print ("Computation time = " + str((toc - tic)) + "s") 
    return subdf




#--------------------------------------------------------------------------------------------------------------------------
#MODEL FUNCTION KERAS
#--------------------------------------------------------------------------------------------------------------------------
#This function builds a model with a certain number of layers, an activation function, an optimizer and the learning rate.

def keras_model_m(layers_dims, learning_rate, a, o):
    L = len(layers_dims)
    
    model = models.Sequential()
    model.add(layers.Dense(layers_dims[1], input_shape=(layers_dims[0],), activation= a ))
    
    for l in range(2, L-1):
        model.add(layers.Dense(layers_dims[l], activation= a, kernel_initializer="random_normal",
                bias_initializer="zeros"))
    
    model.add(layers.Dense(layers_dims[L-1], activation="softmax", kernel_initializer="random_normal",
                bias_initializer="zeros"))
    
    model.compile(optimizer=o(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

#--------------------------------------------------------------------------------------------------------------------------
#BEST MODEL ANN FUNCTION
#--------------------------------------------------------------------------------------------------------------------------
#This function gives us a list with the best architecture and hyperparameters order by its accuracy.

def best_ANN(lay, learning_rate, batch_size, optimizer, optimizer_name, epochs, activation, pxtrain, pytrain, Xtest, ytest, xval, yval):
    tic = timeit.default_timer()
    Lay_results = pd.DataFrame()
    for i in lay:
        for j in learning_rate:
            for bs in batch_size:
                for p, op in enumerate(optimizer):
                    for ep in epochs:
                        for ac in activation:
                            tf.random.set_seed(4)
                            m = keras_model_m(i, j, ac, op)
                            history = m.fit(pxtrain,
                                        pytrain,
                                        verbose=0,
                                        epochs=ep,
                                        batch_size=bs,
                                        validation_data=(xval, yval))

                            score = m.evaluate( Xtest,  ytest, verbose=0)
                            modelos = {'Model': lay.index(i),
                                       'Structure': str(i),
                                       'Learning_rate': j,
                                       'Batch_size': bs,
                                       'Epochs': ep,
                                       'Optimizer':optimizer_name[p],
                                       'Activation': ac,
                                       'Cost': score[0],
                                       'Accuracy': score[1],
                                       }
                            df_element = pd.DataFrame(modelos, index=[0])
                            Lay_results = pd.concat([Lay_results, df_element])
        print(i)
    Lay_results = Lay_results.sort_values('Accuracy', ascending = False).reset_index(drop=True)        
    toc = timeit.default_timer()
    print ("Computation time = " + str((toc - tic)) + "s") 
    return Lay_results



#--------------------------------------------------------------------------------------------------------------------------
#MODEL-EPOCH FUNCTION
#--------------------------------------------------------------------------------------------------------------------------
#This function gives us the accuracy, history and model of the following items

def model_epoch(mi, lr, activation, optimizer, ep, bs, d_class_weights, pxtrain, pytrain, Xtest, ytest, xval, yval):
    tf.random.set_seed(4)
    model = keras_model_m(mi, lr, activation, optimizer)
    history = model.fit(pxtrain,
                        pytrain,
                        verbose=0,
                        epochs=ep,
                        batch_size=bs,
                        class_weight=d_class_weights ,
                        validation_data=(xval, yval))

    score = model.evaluate( Xtest,  ytest, verbose=0)
    return {'loss': score[0], 'accuracy': score[1]}, history, model

#EARLY STOPPING
def model_early(mi, lr, activation, optimizer, early, bs, d_class_weights, pxtrain, pytrain, Xtest, ytest, xval, yval):
    tf.random.set_seed(4)
    model = keras_model_m(mi, lr, activation, optimizer)
    history = model.fit(pxtrain,
                        pytrain,
                        verbose=0,
                        epochs=200,
                        batch_size=bs,
                        class_weight=d_class_weights ,
                        callbacks=[early],
                        validation_data=(xval, yval))

    score = model.evaluate(Xtest, ytest, verbose=0)
    return {'loss': score[0], 'accuracy': score[1]}, history, model


#--------------------------------------------------------------------------------------------------------------------------
#EVOLUTION EPOCHS ANN FUNCTION
#--------------------------------------------------------------------------------------------------------------------------
def plot_prediction(n_epochs, mfit):
    import matplotlib.pyplot as plt 
    plt.style.use("ggplot")
    plt.figure()
    N = n_epochs
    #plt.figure(figsize=(10, 8))
    plt.plot(np.arange(0, N), mfit.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), mfit.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), mfit.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), mfit.history["val_accuracy"], label="val_acc")
    plt.title("Training and Validation | Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.show()
    
    
#EARLY STOPPING
def plot_prediction_early_stopping(mfit):
    import matplotlib.pyplot as plt 
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(mfit.history["loss"], label="train_loss")
    plt.plot(mfit.history["val_loss"], label="val_loss")
    plt.plot(mfit.history["accuracy"], label="train_acc")
    plt.plot(mfit.history["val_accuracy"], label="val_acc")
    plt.title("Training and Validation | Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.show()
    
#--------------------------------------------------------------------------------------------------------------------------
#HEAT MAP CONFUSION MATRIX SIMPLE
#--------------------------------------------------------------------------------------------------------------------------
def heatmapconf(cm, label):
    ax = plt.figure(figsize=(6, 4))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True,fmt='d', ax = ax, cmap = 'YlGnBu', linecolor = 'grey', linewidths =0.03); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels' ,  fontsize = '14', fontweight="bold")
    ax.set_ylabel('True labels',  fontsize = '14', fontweight="bold")
    #ax.set_title('Confusion Matrix', fontsize = '16',fontweight="bold")
    ax.xaxis.set_ticklabels(label,  rotation=0, fontsize = '11')
    ax.yaxis.set_ticklabels(label,  rotation=0, fontsize = '11')
    return
    

#--------------------------------------------------------------------------------------------------------------------------
#HEAT MAP CONFUSION MATRIX + PRECISION + RECALL
#--------------------------------------------------------------------------------------------------------------------------
    
def heatmapconf_recall_prec(cm, label):
    columns = np.apply_along_axis(sum, 0, cm)
    rows = np.apply_along_axis(sum, 1, cm)


    precisioncol=[]
    recallcol=[]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
            precisioncol.append('P:'+ str(round((cm[i,j]/columns[j])*100,1))+'%')
            recallcol.append('R:'+ str(round((cm[i,j]/rows[i])*100,1)) + '%')


    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_counts,precisioncol, recallcol)]
    labels = np.asarray(labels).reshape(8,8)
    
    ax = plt.figure(figsize=(9, 7))
    ax= plt.subplot()
    sns.heatmap(cm, annot=labels,fmt='', ax = ax, cmap = 'YlGnBu', linecolor = 'grey', linewidths =0.03); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels' ,  fontsize = '16', fontweight="bold")
    ax.set_ylabel('True labels',  fontsize = '16', fontweight="bold")
    #ax.set_title('Confusion Matrix', fontsize = '16',fontweight="bold")
    ax.xaxis.set_ticklabels(label,  rotation=0, fontsize = '13')
    ax.yaxis.set_ticklabels(label,  rotation=0, fontsize = '13')
    return 


    


# In[ ]:




