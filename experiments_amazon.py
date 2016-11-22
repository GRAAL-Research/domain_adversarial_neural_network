import numpy as np
from DANN import DANN
from sklearn.datasets import load_svmlight_files
from sklearn import svm


def main():
    data_folder = './data/'     # where the datasets are
    source_name = 'dvd'         # source domain: books, dvd, kitchen, or electronics
    target_name = 'electronics' # traget domain: books, dvd, kitchen, or electronics
    adversarial = False          # set to False to learn a standard NN

    hidden_layer_size = 50
    lambda_adapt = 0.1 if adversarial else 0.
    learning_rate = 0.001
    maxiter = 200

    print("Loading data...")
    xs, ys, xt, _, xtest, ytest = load_amazon(source_name, target_name, data_folder, verbose=True)

    nb_valid = int(0.1 * len(ys))
    xv, yv = xs[-nb_valid:, :], ys[-nb_valid:]
    xs, ys = xs[0:-nb_valid, :], ys[0:-nb_valid]

    print("Fit...")
    algo = DANN(lambda_adapt=lambda_adapt, hidden_layer_size=hidden_layer_size, learning_rate=learning_rate,
                maxiter=maxiter, epsilon_init=None, seed=12342, adversarial_representation=adversarial, verbose=True)
    algo.fit(xs, ys, xt, xv, yv)

    print("Predict...")
    prediction_train = algo.predict(xs)
    prediction_valid = algo.predict(xv)
    prediction_test = algo.predict(xtest)

    print('Training Risk   = %f' % np.mean(prediction_train != ys))
    print('Validation Risk = %f' % np.mean(prediction_valid != yv))
    print('Test Risk       = %f' % np.mean(prediction_test != ytest))

    print('==================================================================')

    print('Computing PAD on DANN representation...')
    pad_dann = compute_proxy_distance(algo.hidden_representation(xs), algo.hidden_representation(xt), verbose=True)
    print('PAD on DANN representation = %f' % pad_dann)

    print('==================================================================')

    print('Computing PAD on original data...')
    pad_original = compute_proxy_distance(xs, xt, verbose=True)
    print('PAD on original data = %f' % pad_original)


def load_amazon(source_name, target_name, data_folder=None, verbose=False):
    """
    Load the amazon sentiment datasets from svmlight format files
    inputs:
        source_name : name of the source dataset
        target_name : name of the target dataset
        data_folder : path to the folder containing the files
    outputs:
        xs : training source data matrix
        ys : training source label vector
        xt : training target data matrix
        yt : training target label vector
        xtest : testing target data matrix
        ytest : testing target label vector
    """

    if data_folder is None:
        data_folder = 'data/'

    source_file = data_folder + source_name + '_train.svmlight'
    target_file = data_folder + target_name + '_train.svmlight'
    test_file = data_folder + target_name + '_test.svmlight'

    if verbose:
        print('source file:', source_file)
        print('target file:', target_file)
        print('test file:  ', test_file)

    xs, ys, xt, yt, xtest, ytest = load_svmlight_files([source_file, target_file, test_file])

    # Convert sparse matrices to numpy 2D array
    xs, xt, xtest = (np.array(X.todense()) for X in (xs, xt, xtest))

    # Convert {-1,1} labels to {0,1} labels
    ys, yt, ytest = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, ytest))

    return xs, ys, xt, yt, xtest, ytest


def compute_proxy_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


if __name__ == '__main__':
    main()

