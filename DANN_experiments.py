import numpy as np
from DANN import DANN
from load_svmlight_data import load_amazon


data_folder = './data/' # where the datasets are
source_name = 'books'   # source domain: books, dvd, kitchen, or electronics
target_name = 'dvd'     # traget domain: books, dvd, kitchen, or electronics

lambda_adapt = 0.001
hidden_layer_size = 20
learning_rate = 0.001
maxiter = 200

print("loading data ...")
xs, ys, xt, _, xtest, ytest = load_amazon(source_name, target_name, data_folder)

nb_valid = int(0.1 * len(ys))
xv, yv = xs[-nb_valid:, :], ys[-nb_valid:]
xs, ys = xs[0:-nb_valid, :], ys[0:-nb_valid]

print("Fit...") 
algo = DANN(lambda_adapt=lambda_adapt, hidden_layer_size=hidden_layer_size, learning_rate=learning_rate, 
            maxiter=maxiter, epsilon_init=None, seed=12342, verbose=True)
algo.fit(xs, ys, xt, xv, yv)

print("Predict...")
prediction_train = algo.predict(xs)
prediction_valid = algo.predict(xv)
prediction_test = algo.predict(xtest)

print('Training Risk = %f' % np.mean(prediction_train != ys))
print('Validation Risk = %f' % np.mean(prediction_valid != yv))
print('Test Risk = %f' % np.mean(prediction_test != ytest))


