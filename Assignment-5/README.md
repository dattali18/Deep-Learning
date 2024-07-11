# Deep Learning
## Assignment 5 Part 2

In this assignment that objective is to get the `VGG` model from `keras`
and perform a **Transfer Learning** operation on the model, and then retrain
the model to classify the Dogs & Cats datasets.

```python
from keras.applications import VGG16
from keras.layers import Flatten

def model():
    # load the model from VGG16
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark the layers as non-trainable
    for layer in model.layers:
        layer.trainable = False
    
    # add a new classifier layer
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(10, activation='softmax')(class1)
    # define the new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile the model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
```