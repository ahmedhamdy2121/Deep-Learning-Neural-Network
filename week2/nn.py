import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from sys import exit
from sys import stdout
from ctypes import windll

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
print (classes)
print ("Training dataset X: " + str(train_set_x_orig.shape))
print ("Training dataset Y: " + str(train_set_y.shape))
print ("Testing dataset X: " + str(test_set_x_orig.shape))
print ("Testing dataset Y: " + str(test_set_y.shape))

# Figure out the dimensions and shapes
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print ("Number of training examples = " + str(m_train))
print ("Number of testing examples = " + str(m_test))
print ("Image size = " + str(num_px) + " x " + str(num_px))

# Reshape the datasets
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print ("Flatten training set: " + str(train_set_x_flatten.shape))
print ("Flatten testing set: " + str(test_set_x_flatten.shape))
stdout.flush()

# Standardize the data
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

# sigmoid
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1. / ( 1 + np.exp(-z))
    
    return s


# initialize_with_zeros 
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

# propagate
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    grads -- dictionary containing:
        dw (gradient of the loss with respect to w, thus same shape as w)
        db (gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1] # number of examples
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,  X) + b) # compute activation
    cost = (-1. / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=1) # compute cost # cols sums
    # Note: np.multiply is equivalent to *

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1. / m) * np.dot(X, ((A - Y).T))
    db = (1. / m) * np.sum(A - Y, axis=1) # cols sum

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost) # before squeeze the shape of z was (1,)
    assert(cost.shape == ()) # validate it is a number!
    
    grads = {"dw": dw,
             "db": db}

    return grads, cost

# optimize & call gradient descent
def optimize(w, b, X, Y, num_iterations, learning_rate, step = 100, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    step -- the step at which cost is recorded into costs list
    print_cost -- True to print the loss every 'step' steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - learning_rate * dw
        b = b -  learning_rate * db

        # Record the costs
        if i % step == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % step == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

# predict
def predict(w, b, X, print_A = False):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    assert (w.shape == (X.shape[0], 1))
    #w = w.reshape(X.shape[0], 1) # we don't need it, it is just to be sure

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b) # compute activation

    # Convert probabilities A[0,i] to actual predictions p[0,i]
    if print_A:
        [print(x) for x in A]
    Y_prediction = (A > 0.5).astype(int)

    assert (Y_prediction.shape == (1, X.shape[1]))

    return Y_prediction

# model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, step = 100, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """

    m = X_train.shape[0]

    # Initialize parameters with zeros
    w, b = initialize_with_zeros(m)

    # Gradient descent
    params, _grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, step, print_cost)
    # Retrieve parameters w and b from dictionary "parameters"
    w = params["w"]
    b = params["b"]

    # Predict test/train set examples
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

### Model Run ###
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
stdout.flush()

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate = " + str(d["learning_rate"]))
plt.show()

# Check the wrong classified 'Test' data
##  Styles:
##  0 : OK
##  1 : OK | Cancel
##  2 : Abort | Retry | Ignore
##  3 : Yes | No | Cancel
##  4 : Yes | No
##  5 : Retry | No 
##  6 : Cancel | Try Again | Continue
##
## Return values:
## OK = 1
## CANCEL = 2
## ABORT = 3
## Retry = 4
## Ignore = 5
## YES = 6
## NO = 7
## Try Again = 10
## continue = 11
answer = windll.user32.MessageBoxW(0, "Show wrong classified images in test dataset?", "Wrong classified images", 3)
if not answer == 2:
    counter = 1
    Y_prediction_test = d['Y_prediction_test']
    for i in range(test_set_x_orig.shape[0]):
        if Y_prediction_test[:, i] != test_set_y[:, i]:
            s = "y = " + str(test_set_y[0, i]) + ", you predicted that it is a \"" + \
                classes[Y_prediction_test[0, i]].decode("utf-8") + "\" picture."
            if answer == 6:
                plt.title(s)
                plt.figure(1).canvas.set_window_title('Faulty Image #' + str(counter))
                plt.imshow(test_set_x_orig[i])
                plt.show()
            print (s)
            counter += 1

# choice of learning rate
answer = windll.user32.MessageBoxW(0, "Run the learning rate comparison?", "Choice of learning rate", 1)
if answer == 1:
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    stdout.flush()

# check external image
answer = windll.user32.MessageBoxW(0, "Try the model with external image?", "External image", 1)
if answer == 1:
    # change this to the name of the image file 
    my_image = "cat1.jpg"

    # preprocess the image to fit the algorithm
    fname = "datasets/" + my_image
    image = np.array(ndimage.imread(fname, flatten = False))
    my_image = scipy.misc.imresize(image, size = (num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
        classes[int(np.squeeze(my_predicted_image)), ].decode("utf-8") + "\" picture.")

### ============================================ Debugging Run ==================================================== ###
def test_run():
    print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
    dim = 2
    w, b = initialize_with_zeros(dim)
    print ("w = " + str(w))
    print ("b = " + str(b))

    w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
    grads, cost = propagate(w, b, X, Y)
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print ("cost = " + str(cost))

    params, grads, costs = optimize(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = True)
    print ("w = " + str(params["w"]))
    print ("b = " + str(params["b"]))
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print ("costs: " + str(costs))

    print ("predictions = " + str(predict(w, b, X, print_A = True)))

def exampleFun():
    # Example of a picture# Example 
    index = 20
    example = train_set_x_orig[index]
    plt.imshow(example)
    plt.show()
    print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(
        train_set_y[:, index])].decode("utf-8") + "' picture.")

# Running examples
#exampleFun()
#test_run()
