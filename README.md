# Single Hiddel Layer Neural-Network To Find Best Performing Activation Function

Report: Learning Activations in Neural Networks

neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process 
that mimics the way the human brain operates. ANNs are composed of multiple nodes,The neurons are connected by links and they 
interact with each other The nodes can take input data and perform simple operations on the data The result of these operations 
is passed to other neurons. The output at each node is called its activation or node value.

Our created Neural Network consist only one hidden layer that accepts the inputs,weights and activation function and with the
help of these try to provide output by performaing Back Propagation and Forward Propagation.

Code Explanation - 
The Algorithm to find best Activation Function by the Neural Network consist multiple functions.
 1. load_dataset(): MNIST dataset is used to perform classification. Here we are loading and dividing the 
    MNIST dataset to train and test part for trainning and testing.
    Here we are building the input vector from the 28x28 pixels i.e 784
     X_train = X_train.reshape(60000, 784)
     X_test = X_test.reshape(10000, 784)
     X_train = X_train.astype('float32')
     X_test = X_test.astype('float32') 
     X_train /= 255
     X_test /= 255
 
 2. prep_pixels(y_train, y_test): scaling the target vector pixels into binary vectors using One Hot Encoding technique.
    This gives a 10 classified classes i.e (0-9).

 3. define_model(af): Defining and compiling the single hidden layer neural network model using keras library. Providing
    the input_shape,units(output multiclass-classification) and activation function. for compiling the model,loss='categorical_crossentropy',
    metrics=['accuracy'], optimizer='adam' parameters are defined.
 
 4. train_model(af): Trainning and fitting the model with X_train,Y_train and collecting the model history for accuracy, loss, val_loss and val_accuracy.

 5. get_best_af(): Providing the neural network with activation functions for training and selecting the best activation function for the model.
    function iterates over the multiple activation functions to get loss, accuracy,val_loss and val_accuracy. Finally the best activation function with less loss 
    and high accuracy is selected for the final model. plotting of the loss function vs epochs.
 
 6. train_final_model(): The algorithm provides the best activation function by exploiting a flexible function. and final model is trained and evaluated using 
    best activation function.
 
 7. From the observations we can conclude that softmax provides better accuracy and less loss during training and evaluating the model.
    Test Loss 0.2796614468097687
    Test Accuracy 0.9223999977111816
    f1_score 0.9223876185408042
    Model 9224 times classified correctly and 776 times classified incorrectly
    The detailed classification report is implemented in code with respective of Recall and Precision.
