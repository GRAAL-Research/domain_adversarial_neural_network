import numpy as np
from DANN import DANN
from load_data import load_representations

    
context_folder = './data'
dataset_name = 'books.dvd'
print "loading data ..."
xs, ys = load_representations(context_folder, dataset_name, noise = 0., suffix = 's')
xt, _ = load_representations(context_folder, dataset_name, noise = 0., suffix = 't')
xtest, ytest = load_representations(context_folder,dataset_name, noise = 0., suffix = 'test') 
ys = (ys + 1)/2
ytest = (ytest+1)/2
nb_valid = int(0.1*len(ys)) 
xv, yv = xs[-nb_valid:,:], ys[-nb_valid:]            
xs, ys = xs[0:-nb_valid,:], ys[0:-nb_valid]

lambda_adapt = 0.001
hidden_layer_size = 100
learning_rate = 0.001
maxiter = 200
print "Fit..." 
algo = DANN(lambda_adapt = lambda_adapt, hidden_layer_size = hidden_layer_size, learning_rate = learning_rate, maxiter = maxiter,  epsilon_init = None, seed = 12342)
algo.fit(xs, ys, xt, xv, yv)
print "Predict..." 
prediction_train = algo.predict(xs)
prediction_valid = algo.predict(xv)
prediction_test = algo.predict(xtest)

print 'Training Risk' , np.mean(prediction_train != ys)
print 'Validation Risk' , np.mean(prediction_valid != yv)
print 'Test Risk' , np.mean(prediction_test != ytest)


