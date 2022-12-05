
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
import cv2
import glob
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.utils


print("Reading Data from folders!")
data = []
dataset_x=[]
dataset_y=[]

for filename in glob.glob('./positive/*.png'):        
    im=cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(im, (128,128),interpolation = cv2.INTER_AREA)
    dataset_x.append(np.reshape(resized, [128,128,1]))
    dataset_y.append(1)

for filename in glob.glob('./negative/*.png'):        
    im=cv2.imread(filename)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, (128,128),interpolation = cv2.INTER_AREA)
    dataset_x.append(np.reshape(resized, [128,128,1]))
    dataset_y.append(0)

# converting in ARRAY
dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)
    
#splitting the dataset into 80% training and 20% teting 
X_train,X_test,Y_train,Y_test=train_test_split(dataset_x,dataset_y,train_size=0.8,test_size=0.2)

# Change the labels from categorical to one-hot encoding
Y_train = to_categorical(Y_train, num_classes = 2)
Y_test = to_categorical(Y_test, num_classes = 2)

batch_size = 32                    
num_classes = 2 
model = Sequential()
    
model.add(Conv2D(filters =32, kernel_size=(3, 3),activation ='relu',input_shape=(128,128,1),padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
    
model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation ='relu',padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
    
model.add(Conv2D(filters =128,kernel_size= (3, 3), activation ='relu',padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
    
# NORMALIZATION

model.add(Flatten())
model.add(Dense(1024, activation = "relu",kernel_initializer = keras.initializers.he_normal(seed=None)))
model.add(Dropout(0.5))
model.add(Dense(512, activation = "relu" , kernel_initializer = keras.initializers.he_normal(seed=None)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

print("### Please wait to run the entire program ###")



''' Adam Model is as follows'''

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        
print("### Training Adam Model ###")
        
train = model.fit(X_train, Y_train, batch_size=batch_size,epochs=15,verbose=1,validation_data=(X_test, Y_test))
        
print("...................Testing......................")
        
test_eval = model.evaluate(X_test, Y_test, verbose=0)
    
print(".............ADAM MODEL'S ACCURACY...................")  
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
print("...............................................................................................................")
    
    
    

''' SGD Model is as follows'''


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),metrics=['accuracy'])
        
print("...........................Training SGD MODEL............................")
        
train = model.fit(X_train, Y_train, batch_size=batch_size,epochs=15,verbose=1,validation_data=(X_test, Y_test))
        
print("...............Testing..............")
        
test_eval = model.evaluate(X_test, Y_test, verbose=0)
      
print(".......................SGD Optimizer's accuracy...........................")  
      
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
    
print("......................................................................................................")
    
    

    

'''RMS Model'''
  

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9),metrics=['accuracy'])
        
print("-----------------------Training RMS Model------------------------------------------")
        
train = model.fit(X_train, Y_train, batch_size=batch_size,epochs=15,verbose=1,validation_data=(X_test, Y_test))
        
print("-----------------------Testing------------------------------------------")
        
test_eval = model.evaluate(X_test, Y_test, verbose=0)
        
print("----------------------- RMSprop's accuracy------------------------------------------")
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
    
print("--------------------------------------------------------------------------------------")
    
    


