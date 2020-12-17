import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1/255)

training_set = train_datagen.flow_from_directory('bird/train',
                                                 target_size = (128,128),
                                                 color_mode = 'rgb',
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('bird/test',
                                                 target_size = (128,128),
                                                 color_mode = 'rgb',
                                                 batch_size = 32,
                                                 class_mode = 'binary')

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
    
classifier = Sequential()

classifier.add(Convolution2D(32, (3,3), input_shape = (128,128,3), activation= 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())


classifier.add(Dense(units = 4000,kernel_initializer = 'uniform',activation ='relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 1000,kernel_initializer = 'uniform',activation ='relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 500,kernel_initializer = 'uniform',activation ='relu'))

classifier.add(Dense(units = 3,kernel_initializer = 'uniform',activation ='softmax'))

classifier.summary()
classifier.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

classifier.fit_generator(training_set,
                         samples_per_epoch = 500,
                         epochs = 70)

pred = classifier.evaluate_generator(generator=test_set,steps=4.5)

training_set.class_indices

pred1 = classifier.predict_generator(test_set,steps=4.5) 