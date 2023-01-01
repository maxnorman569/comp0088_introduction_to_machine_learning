#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COMP0088 lab exercises for week 3.

Add your code as specified below.

A simple test driver is included in this script. Call it at the command line like this:

  $ python week_3.py

A 4-panel figure, `week_3.pdf`, will be generated so you can check it's doing what you
want. You should not need to edit the driver code, though you can if you wish.
"""

import sys, os, os.path
import argparse
import pprint

import numpy as np
import numpy.random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import utils


#### ADD YOUR CODE BELOW

# -- Question 1 --

# - Auxiliary Functions -

# Returns euclidean distance between two points
def euclidean_distance(point_1, point_2):
    dist = np.linalg.norm(point_1 - point_2)
    return dist

# Returns class with most number of votes
def vote(points):
    classes, counts = np.unique(points, return_counts=True)
    vote = classes[np.argmax(counts)]
    return vote

    
def nearest_neighbours_predict ( train_X, train_y, test_X, neighbours=1 ):
    """
    Predict labels for test data based on neighbourhood in
    training set.
    
    # Arguments:
        train_X: an array of sample data for training, where rows
            are samples and columns are features.
        train_y: vector of class labels corresponding to the training
            samples, must be same length as number of rows in X
        test_X: an array of sample data to generate predictions for,
            in same layout as train_X.
        neighbours: how many neighbours to canvass at each test point
        
    # Returns
        test_y: predicted labels for the samples in test_X
    """
    assert(train_X.shape[0] == train_y.shape[0])
    assert(train_X.shape[1] == test_X.shape[1])

    test_y = np.zeros(test_X.shape[0])

    for i, x_test_point in enumerate(test_X):
        distances = [ (euclidean_distance(x_test_point, x), y) for x, y in zip(train_X, train_y)]
        distances.sort(key=lambda z: z[0]) # sorts tuples based on first entry of tuple
        test_y[i] += vote( [z[1] for z in distances[:neighbours]] ) # casts vote

    return test_y


# -- Question 2 --

def misclassification ( y, cls, weights=None ):
    """
    Calculate (optionally-weighted) misclassification error for
    a given set of labels if assigned the given class.
    
    # Arguments
        y: a set of class labels
        cls: a candidate classification for the set
        weights: optional weights vector specifying relative
            importance of the samples labelled by y
    
    # Returns
        err: the misclassification error of the candidate labels
    """
    if weights is None:
        return np.sum((1 / len(y)) * np.where(y != cls, 1, 0))
    else:
        return np.sum(weights * np.where(y != cls, 1, 0))


def decision_node_split ( X, y, cls=None, weights=None, min_size=3 ):
    """
    Find (by brute force) a split point that best improves the weighted
    misclassification error rate compared to the original one (or not, if
    there is no improvement possible).
    
    Features are assumed to be numeric and the test condition is
    greater-or-equal.
    
    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of class labels corresponding to the samples,
            must be same length as number of rows in X
        cls: class label currently assigned to the whole set
            (if not specified we use the most common class in y, or
            the lowest such if 2 or more classes occur equally)
        weights: optional weights vector specifying relevant importance
            of the samples
        min_size: don't create child nodes smaller than this
    
    # Returns:
        feature: index of the feature to test (or None, if no split)
        thresh: value of the feature to test (or None, if no split)
        c0: class assigned to the set with feature < thresh
            (or None, if no split)
        c1: class assigned to the set with feature >= thresh
            (or None, if no split)
    """
    assert(X.shape[0] == len(y))

    # stopping criteria - prevents getting one node / datapoint
    if len(y) < min_size * 2:
        return None, None, None, None

    # initial classification
    if cls is None: 
        cls = vote(y) 

    # if no weights provided, weigh all points equally
    if weights is None:
        weights = np.ones(len(y))/len(y)

    # calculate the current missclassification error
    error = misclassification(y, cls=cls, weights=weights)

    # if node is pure - stop
    if error == 0: 
        return None, None, None, None

    # criterion
    best_feat, best_thresh = None, None
    best_c0, best_c1 = None, None
    best_improvement = 0
    
    for feat in range(X.shape[-1]):
        for thresh in X[:,feat]:
            # split masks
            mask_0 =  X[:,feat] >= thresh
            mask_1 = ~mask_0

            # if split is smaller than min_size continue to next iteration
            if (np.sum(mask_0) < min_size) or (np.sum(mask_1) < min_size):
                continue

            # get class labels corresponding to split
            y_0 = y[mask_0]
            y_1 = y[mask_1]

            # get weights corresponding to split
            w_0 = weights[mask_0]
            w_1 = weights[mask_1]

            # get unique classes in each split
            classes_0 = np.unique(y_0)
            classes_1 = np.unique(y_1)

            # get missclassification rate for each class in each split
            errors_0 = [misclassification(y_0, cls=class_, weights=w_0) for class_ in classes_0]
            errors_1 = [misclassification(y_1, cls=class_, weights=w_1) for class_ in classes_1]

            # set predicted class at the class with the minimum missclassification error
            class_0 = classes_0[np.argmin(errors_0)]
            class_1 = classes_1[np.argmin(errors_1)]

            # set the corresponding missclassification errors for the predicted class
            error_0 = np.min(errors_0)
            error_1 = np.min(errors_1)

            # get the imporovement of this split on the previous missclassification error
            improvement = error - (error_0 + error_1)

            # compare the improvement to previous splits
            if improvement > best_improvement:
                best_feat = feat
                best_thresh = thresh
                best_improvement = improvement
                best_c0 = class_0
                best_c1 = class_1

    # if there is no improvement for any split in any feature, stop
    if best_feat is None:
        return None, None, None, None
    
    return best_feat, best_thresh, best_c0, best_c1


def decision_tree_train ( X, y, cls=None, weights=None,
                          min_size=3, depth=0, max_depth=10 ):
    """
    Recursively choose split points for a training dataset
    until no further improvement occurs.
    
    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of class labels corresponding to the samples,
            must be same length as number of rows in X
        cls: class label currently assigned to the whole set
            (if not specified we use the most common class in y, or
            the lowest such if 2 or more classes occur equally)
        weights: optional weights vector specifying relevant importance
            of the samples
        min_size: don't create child nodes smaller than this
        depth: current recursion depth
        max_depth: maximum allowed recursion depth
    
    # Returns:
        tree: a dict containing (some of) the following keys:
            'kind' : either 'leaf' or 'decision'
            'class' : the class assigned to this node (leaf)
            'feature' : index of feature on which to split (decision)
            'thresh' : threshold at which to split the feature (decision)
            'below' : a nested tree for when feature < thresh (decision)
            'above' : a nested tree for when feature >= thresh (decision)
    """
    if cls is None: 
        cls = vote(y)
        
    if depth == max_depth:
        return { 'kind' : 'leaf', 'class' : cls }
    
    feat, thresh, cls0, cls1 = decision_node_split ( X, y, cls=cls, weights=weights, min_size=min_size )
    
    if feat is None:
        return { 'kind' : 'leaf', 'class' : cls }
    
    mask_1 = X[:,feat] >= thresh
    mask_2 = ~mask_1
    
    return { 'kind' : 'decision',
             'feature' : feat,
             'thresh' : thresh,
             'above' : decision_tree_train(X[mask_1,:], y[mask_1], cls1, None if weights is None else weights[mask_1], min_size, depth+1, max_depth),
             'below' : decision_tree_train(X[mask_2,:], y[mask_2], cls0, None if weights is None else weights[mask_2], min_size, depth+1, max_depth) }


def decision_tree_predict1 ( tree, x ):
    while True:
        if tree['kind'] == 'leaf':
            return tree['class']
        
        tree = tree['above'] if x[tree['feature']] >= tree['thresh'] else tree['below']
    

def decision_tree_predict ( tree, X ):
    """
    Predict labels for test data using a fitted decision tree.
    
    # Arguments
        tree: a decision tree dictionary returned by decision_tree_train
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: the predicted labels
    """

    return np.array([ decision_tree_predict1( tree, X[ii,:] ) for ii in range(X.shape[0]) ])


# -- Question 3 --

def random_forest_train ( X, y, k, rng, min_size=3, max_depth=10 ):
    """
    Train a (simplified) random forest of decision trees.
    
    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of binary class labels corresponding to the
            samples, must be same length as number of rows in X
        k: the number of trees in the forest
        rng: an instance of numpy.random.Generator
            from which to draw random numbers        
        min_size: don't create child nodes smaller than this
        max_depth: maximum tree depth
    
    # Returns:
        forest: a list of tree dicts as returned by decision_tree_train
    """
    
    # list of decision trees
    forest = []

    for i in range(k):
        # bootstrapped data
        boosted_i = rng.choice(X.shape[0], X.shape[0]) 
        X_i = X[boosted_i,:]
        y_i = y[boosted_i]
        forest.append(decision_tree_train ( X_i, y_i, min_size=min_size, max_depth=max_depth ))
    
    return forest
    

def random_forest_predict ( forest, X ):
    """
    Predict labels for test data using a fitted random
    forest of decision trees.
    
    # Arguments
        forest: a list of decision tree dicts
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: the predicted labels
    """
    y_preds = np.array([ decision_tree_predict( tree, X ) for tree in forest ])
    return np.array([ vote(y_preds[:,i]) for i in range(y_preds.shape[1])])


# -- Question 4 --

def adaboost_train ( X, y, k, min_size=1, max_depth=1, epsilon=1e-8 ):
    """
    Iteratively train a set of decision tree classifiers
    using AdaBoost.
    
    # Arguments:
        X: an array of sample data, where rows are samples
            and columns are features.
        y: vector of binary class labels corresponding to the
            samples, must be same length as number of rows in X
        k: the maximum number of weak classifiers to train
        min_size: don't create child nodes smaller than this
        max_depth: maximum tree depth -- by default we just
            use decision stumps
        epsilon: threshold below which the error is considered 0
    
    # Returns:
        trees: a list of tree dicts as returned by decision_tree_train
        alphas: a vector of weights indicating how much credence to
            given each of the decision tree predictions
    """
    weights = np.ones(X.shape[0])/X.shape[0]
    alphas = []
    trees = []
    
    for ii in range(k):
        trees.append(decision_tree_train(X, y, weights=weights, min_size=min_size, max_depth=max_depth))
        pred_y = decision_tree_predict(trees[-1], X)
        err = np.dot(weights, pred_y != y)
        
        # bail if we're classifying perfectly
        if err < epsilon:
            alphas.append(1)
            break
        
        alphas.append(np.log((1 - err)/err))
        
        weights = weights * np.exp(alphas[-1] * (pred_y != y))
        weights = weights / np.sum(weights)
    
    return trees, np.array(alphas)


def adaboost_predict ( trees, alphas, X ):
    """
    Predict labels for test data using a fitted AdaBoost
    ensemble of decision trees.
    
    # Arguments
        trees: a list of decision tree dicts
        alphas: a vector of weights for the trees
        X: an array of sample data, where rows are samples
            and columns are features.

    # Returns
        y: the predicted labels
    """
    preds = np.array([ decision_tree_predict( tree, X ) for tree in trees ]).T * 2 - 1
    weighted = preds @ alphas
    
    return (weighted >= 0).astype(int)
    

#### TEST DRIVER

def process_args():
    ap = argparse.ArgumentParser(description='week 3 coursework script for COMP0088')
    ap.add_argument('-s', '--seed', help='seed random number generator', type=int, default=None)
    ap.add_argument('-n', '--num_samples', help='number of samples to use', type=int, default=50)
    ap.add_argument('-k', '--neighbours', help='number of neighbours for k-NN fit', type=int, default=3)
    ap.add_argument('-m', '--min_size', help='smallest acceptable tree node', type=int, default=3)
    ap.add_argument('-w', '--weak', help='how many weak classifiers to train for AdaBoost', type=int, default=10)
    ap.add_argument('-f', '--forest', help='how many trees to train for random forest', type=int, default=10)
    ap.add_argument('-r', '--resolution', help='grid sampling resolution for classification plots', type=int, default=20)
    ap.add_argument('-d', '--data', help='CSV file containing training data', default='week_3_data.csv')
    ap.add_argument('file', help='name of output file to produce', nargs='?', default='week_3.pdf')
    return ap.parse_args()

if __name__ == '__main__':
    args = process_args()
    rng = numpy.random.default_rng(args.seed)
    
    print(f'loading data from {args.data}')
    df = pd.read_csv(args.data)
    X = df[['X1','X2']].values[:args.num_samples,:]
    y = df['Multi'].values[:args.num_samples]

    fig = plt.figure(figsize=(10, 10))
    axs = fig.subplots(nrows=2, ncols=2)
    
    print(f'Q1: checking {args.neighbours}-nearest neighbours fit')
    # this is a fudge -- there's no training phase, so check implementation with a dummy prediction
    dummy = nearest_neighbours_predict ( X[:2,:], y[:2], X[:2,:], neighbours=args.neighbours )
    if dummy is None:
        print('decision tree not implemented')
        utils.plot_unimplemented(axs[0,0], f'{args.neighbours}-Nearest Neighbours')
    else:    
        print(f'Q1: plotting {args.neighbours}-nearest neighbours fit')    
        nn_cls = lambda z: nearest_neighbours_predict ( X, y, z, neighbours=args.neighbours )
        utils.plot_classification_map(axs[0,0], nn_cls, X, y, resolution=args.resolution, title=f'{args.neighbours}-Nearest Neighbours')
    
    print('Q2: testing misclassification error')
    all_right = misclassification(np.ones(3), 1)
    all_wrong = misclassification(np.ones(3), 0)
    fifty_fifty = misclassification(np.concatenate((np.ones(3), np.zeros(3))), 1)
    
    right_msg = 'correct' if np.isclose(all_right, 0) else 'wrong, should be 0'
    wrong_msg = 'correct' if np.isclose(all_wrong, 1) else 'wrong, should be 1'
    fifty_msg = 'correct' if np.isclose(fifty_fifty, 0.5) else 'wrong should b 0.5'
    
    print(f' all right: {all_right} - {right_msg}')
    print(f' all wrong: {all_wrong} - {wrong_msg}')
    print(f' fifty-fifty: {fifty_fifty} - {fifty_msg}')
    
    print('Q2: fitting decision tree')
    tree = decision_tree_train ( X, y, min_size=args.min_size )
    
    if tree is None:
        print('decision tree not implemented')
        utils.plot_unimplemented(axs[0,1], 'Decision Tree')
    else:
        print('Q2: plotting decision tree fit')
        tree_cls = lambda z: decision_tree_predict ( tree, z )
        utils.plot_classification_map(axs[0,1], tree_cls, X, y, resolution=args.resolution, title='Decision Tree')
    
    print(f'Q3: fitting random forest with {args.forest} trees')
    forest = random_forest_train ( X, y, args.forest, rng=rng, min_size=args.min_size )
    
    if forest is None:
        print('random forest not implemented')
        utils.plot_unimplemented(axs[1,0], 'Random Forest')
    else:
        print('Q3: plotting random forest fit')
        forest_cls = lambda z: random_forest_predict ( forest, z )
        utils.plot_classification_map(axs[1,0], forest_cls, X, y, resolution=args.resolution, title=f'Random Forest ({args.forest} Trees)')
        
    print('Q4: fitting adaboost ensemble')
    # swap to binary labels since we're only doing 2-class AdaBoost
    y = df['Binary'].values[:args.num_samples]
    trees, alphas = adaboost_train ( X, y, args.weak )
    
    if trees is None:
        print('adaboost not implemented')
        utils.plot_unimplemented(axs[1,1], 'AdaBoost')
    else:   
        print('Q4: plotting AdaBoost fit')
        ada_cls = lambda z: adaboost_predict ( trees, alphas, z )
        utils.plot_classification_map(axs[1,1], ada_cls, X, y, resolution=args.resolution, title=f'AdaBoost ({args.weak} Stumps)')

    fig.tight_layout(pad=1)
    fig.savefig(args.file)
    plt.close(fig)
