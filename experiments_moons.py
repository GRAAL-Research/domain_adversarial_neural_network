import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from DANN import DANN

from matplotlib import pyplot
from scipy.spatial.distance import cdist


def main():
    X, y, Xt, yt = make_trans_moons(35, nb=150)
    Xall = np.vstack((X, Xt))

    special = np.array([[-2.4, -1.6], [-1.2, 0.4], [.8, -.5], [2.5, 1.5]])
    special_points = Xall[np.argmin(cdist(special, Xall), axis=1), :]

    # Standard NN
    algo = DANN(hidden_layer_size=15, maxiter=500, lambda_adapt=6., seed=42, adversarial_representation=False)
    algo.fit(X, y, Xt)

    pyplot.subplot(2, 4, 1)
    pyplot.title("NN: Label classification")
    draw_trans_data(X, y, Xt, algo.predict, special_points=special_points,
                    special_xytext=[(40, -15), (-30, -80), (-50, 40), (-70, 0)])

    pyplot.subplot(2, 4, 2)
    pyplot.title("NN: Representation PCA")
    run_pca(X, y, Xt, algo, special_points=special_points, mult=[-1, -1])

    pyplot.subplot(2, 4, 3)
    pyplot.title("NN: Domain classification")
    draw_trans_data(X, y, Xt, algo.predict_domain, colormap_index=1)

    pyplot.subplot(2, 4, 4)
    pyplot.title("NN: Hidden neurons")
    draw_trans_data(X, y, Xt, neurons_to_draw=(algo.W, algo.b))

    # DANN
    algo = DANN(hidden_layer_size=15, maxiter=500, lambda_adapt=6., seed=42)
    algo.fit(X, y, Xt)

    pyplot.subplot(2, 4, 5)
    pyplot.title("NN: Label classification")
    draw_trans_data(X, y, Xt, algo.predict,  special_points=special_points,
                    special_xytext=[(50,-15), (-20,-90), (-50,40), (-80,0)] )

    pyplot.subplot(2, 4, 6)
    pyplot.title("NN: Representation PCA")
    run_pca(X, y, Xt, algo, special_points=special_points, mult=[-1,1],
            special_xytext=[(-10,-80), (50,-60), (-40,50), (-20,70)])

    pyplot.subplot(2, 4, 7)
    pyplot.title("NN: Domain classification")
    draw_trans_data(X, y, Xt, algo.predict_domain, colormap_index=1)

    pyplot.subplot(2, 4, 8)
    pyplot.title("NN: Hidden neurons")
    draw_trans_data(X, y, Xt, neurons_to_draw=(algo.W, algo.b))

    pyplot.show()


def make_trans_moons(theta=40, nb=100, noise=.05):
    from math import cos, sin, pi
    
    X, y = make_moons(nb, noise=noise, random_state=1) 
    Xt, yt = make_moons(nb, noise=noise, random_state=2)
    
    trans = -np.mean(X, axis=0) 
    X  = 2*(X+trans)
    Xt = 2*(Xt+trans)
    
    theta = -theta*pi/180
    rotation = np.array( [  [cos(theta), sin(theta)], [-sin(theta), cos(theta)] ] )
    Xt = np.dot(Xt, rotation.T)
    
    return X, y, Xt, yt


def draw_trans_data(X, y, Xt, predict_fct=None, neurons_to_draw=None, colormap_index=0, special_points=None, special_xytext=None):
    # Some line of codes come from: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    if colormap_index==0:
        cm_bright = ListedColormap(['#FF0000', '#00FF00'])
    else:
        cm_bright = ListedColormap(['#0000FF', '#000000'])

    x_min, x_max = 1.1*X[:, 0].min(), 1.1*X[:, 0].max()
    y_min, y_max = 1.5*X[:, 1].min(), 1.5*X[:, 1].max()
        
    pyplot.xlim((x_min,x_max))
    pyplot.ylim((y_min,y_max))
    
    pyplot.tick_params(direction='in', labelleft=False)    

    if predict_fct is not None:
        h = .02  # step size in the mesh

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = predict_fct(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        pyplot.contourf(xx, yy, Z, cmap=cm_bright, alpha=.4)
        pyplot.contour(xx, yy, Z, colors='black', linewidths=5)

    if X is not None:
        for i in range(len(y)):
            if y[i] == 1:
                pyplot.annotate("-", X[i,:], color="green", size=50*1.5, textcoords='offset points', xytext=(-6*1.5,-13*1.5))
            else:
                pyplot.annotate("+", X[i,:], color="red", size=30*1.5, textcoords='offset points', xytext=(-8*1.5,-8*1.5))

    if Xt is not None:
        pyplot.scatter(Xt[:, 0], Xt[:, 1], c='k', s=40)

    if special_points is not None:
        for i in range(np.shape(special_points)[0]):
            if special_xytext is None:
                xytext = (30,45) if i%2 == 1 else (-40,-60)
            else:
                xytext = special_xytext[i]
                
            pyplot.annotate('ABCDEFG'[i], special_points[i,:], xycoords='data', color="blue",
                xytext=xytext, textcoords='offset points',
                size=32,
                arrowprops=dict(arrowstyle="fancy", fc=(0., 0., 1.), ec="none", connectionstyle="arc3,rad=0.0"))
        
    if neurons_to_draw is not None:
        for w12, b in zip(neurons_to_draw[0], neurons_to_draw[1]):
            w1, w2 = w12
            get_y = lambda x: -(w1*x+b)/w2
            pyplot.plot([x_min,x_max], [get_y(x_min), get_y(x_max)])


def run_pca(X, y, Xt, algo, special_points=None, special_xytext=None, mult=None):
    if mult is None: # mult is used to flip the representation
        mult = np.ones(2)

    h_X = algo.hidden_representation(X)
    h_Xt = algo.hidden_representation(Xt)

    pca = PCA(n_components=2)
    pca.fit(np.vstack((h_X, h_Xt)))

    if special_points is not None:
        special_points = mult*pca.transform(algo.hidden_representation(special_points))

    draw_trans_data(mult*pca.transform(h_X), y, mult*pca.transform(h_Xt), special_points=special_points,
                    special_xytext=special_xytext)


if __name__ == '__main__':
    main()
        
        

