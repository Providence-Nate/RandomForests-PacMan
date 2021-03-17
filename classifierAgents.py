# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print " ###################################################################################################### "
        print "Initialising"
        print "Starting Classifier Agent!"
        print "We are using Random Forests as our classification algorithm!"
        print " ###################################################################################################### "

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):
        print "Running registerInitialState for ClassifierAgent!"

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of integers 0-3 indicating the action
        # taken in that state.

        self.r_target = np.reshape(self.target, (-1,1))
        # Concatenating both the feature values and the target label into one array for ease of data handling.
        self.combined_data = np.concatenate((np.array(self.data), self.r_target), axis=1)

    ###########################################################################################################
    # ####################################################################################################### #
    # ------------------------------ Codes for Random Forest Begins Here ----------------------------------- #
    ###########################################################################################################
    # ####################################################################################################### #

    # Readme: A custom, sophisticated random forest classifier was designed and coded from scratch deliberately for
    # this task. The following functions are all the building blocks for this random forest classifier, including
    # but not limited to functions for calculating entropy and information gain, and functions for the
    # decision tree algorithms etc.

    def purity_check(self, data):

        #
        # Function conducts purity checks to see if one node contains all instances from one label.
        # If this node is pure, then we shall not have to continue with further entropy and purity calculations.
        # Returns true is the data is completely pure and false otherwise.
        #

        # getting column labels for each instance
        column_labels = data[:, -1]
        unique_labels = np.unique(column_labels)

        # if split is 100% pure then return true, else false.
        if len(unique_labels) == 1:
            return True
        else:
            return False

    def class_label(self, data):

        #
        # This function returns the class labels if:
        # 1. The node is completely pure or
        # 2. If we terminate the tree earlier according to some maximum tree depth conditions etc. Then we will have to
        # compute the majority class within the node/leaf of the tree and return that class.
        #

        labels = data[:, -1]
        unique_label, label_frequencies = np.unique(labels, return_counts=True)

        # We will use argmax function to return the unique label that corresponds to the majority class within this
        # node, i.e. label that occurs the most frequently within this node.
        class_index = label_frequencies.argmax()
        classification = unique_label[class_index]
        return classification

    def get_split(self, data, split_attributes, feature_values):

        #
        # Function to split the data according to the attributes we get from get_attributes, and what specific values
        # of these attributes that we should conduct the split on.
        # Note: our values for our features from good-moves.txt is only either 0 or 1. So we split attribute values
        # according to whether they are 0s or 1s.
        # Function returns 2, 2-D numpy array as a result of splitting this data, i.e. left and right child nodes.
        #

        # values across all features for this particular attribute
        column_values = data[:, split_attributes]
        # if column values is less than 0, i.e. it means it is 0, then these instances gets allocated into one node, and
        # the remaining instances which has value of 1 for this particular column gets allocated into another node.
        data_left_node = data[column_values < feature_values]
        data_right_node = data[column_values >= feature_values]
        return data_left_node, data_right_node

    def get_attributes(self, data, k):

        #
        # Function creates a random subset of attributes/features as inputs to our decision trees.
        # For example if our dataset comes with 25 features, we might choose to only sample 'k' features out of
        # those 25 features as input to our trees. k corresponds to the number of random attributes we are
        # going to sample.
        #

        # Creating a dictionary for potential sampled attributes and their corresponding values.
        # The values of these attributes will potential be used for splitting instances at each branch.
        potential_attributes_splits = {}
        _, n_attributes = np.shape(data)
        attribute_indices = [i for i in range(n_attributes-1)]

        # Random sampling to get the attributes. Only works if value of k is less than or equal to the number of
        # features in the dataset. if k isn't true, then we return the entire set of attributes.
        if k and k <= len(attribute_indices):
            attribute_indices = random.sample(population=attribute_indices, k=k)

        # Sample which attributes we wish to put in the potential_attributes dict, along with the unique values across
        # all instances for these particular sampled attributes.
        for index in attribute_indices:
            sampled_values = data[:, index]
            # potential values of the attributes that we can use to split the instances.
            decision_values = np.unique(sampled_values)
            potential_attributes_splits[index] = decision_values
        return potential_attributes_splits

    def entropy_calc(self, data):

        #
        # Function for calculating entropy
        #

        column_labels = data[:, -1]
        # Counting the frequency of each unique label
        _, label_frequency = np.unique(column_labels, return_counts=True)
        label_frequency = label_frequency.astype(float)
        # Calculating probability of each unique label. We get an array of probabilities corresponding to each class
        p = label_frequency/label_frequency.sum()
        # Formula for entropy. Note that numpy has an element wise operation, so this following function of entropy
        # will calculate the probability multiplied by the log2 of each probability corresponding to each element
        # in the 'p' array. We then sum the elements within the array.
        entropy = sum(p * -np.log2(p))
        return entropy

    def entropy_of_split(self, data_left_node, data_right_node):

        # To calculate the overall entropy of a split, we need to consider the total number of features
        # in the parent nodes and the proportion of number of features in each resulting child node.

        # Total number of instances before split.
        total_instances = float(len(data_left_node)+len(data_right_node))
        # Proportion of instances in the left child node to the total number of instances in the parent node
        p_left_node = float(len(data_left_node))/total_instances
        # Proportion of instances in the right child node to the total number of instances in the parent node
        p_right_node = float(len(data_right_node))/total_instances

        # Entropy of a split is just the proportion of instances within each child node to the number of instances in
        # the parent node multiplied by the corresponding entropy of each child node.
        overall_entropy = (p_left_node*self.entropy_calc(data_left_node)) + \
                          (p_right_node*self.entropy_calc(data_right_node))
        return overall_entropy

    def get_best_split(self, data, sampled_attributes):

        #
        # Function that, finds the best split that results in the highest information gain out of all the
        # potential splits. Function returns the parameters of the best split, i.e. which attributes we decide to
        # split the instances on and the corresponding attribute values we decide to split the instances with.
        #

        # We use the get_attributes function above that randomly samples k attributes from the total number of
        # attributes in the dataset.
        split_attributes = sampled_attributes

        # Initializing an arbitrarily high value as an comparison value for the loop below.
        # This will serve as the entropy of the parent node.
        parent_entropy = 1111

        for attribute_index in split_attributes:
            for decision_values in split_attributes[attribute_index]:  # accessing values of a dictionary
                # calling the get_split function above to split the instances into two nodes.
                left_node, right_node = self.get_split(data, attribute_index, decision_values)
                total_children_entropy = self.entropy_of_split(left_node, right_node)

                # If there is information gain, we will store the split that produces this information gain.
                # We will loop through the attributes and attribute values in the set of sampled attributes
                # in order to see which combination of attribute and values produces the highest information gain.
                if total_children_entropy < parent_entropy:
                    parent_entropy = total_children_entropy
                    # split_on is the attribute that we choose to conduct the split on, and split_attribute_value
                    # corresponds to the value that we will split our instances with.
                    split_on, split_attribute_value = attribute_index, decision_values
        return split_on, split_attribute_value

    def decision_trees(self, data, depth_count=0, min_instances=15, max_tree_depth=5, number_features=None):

        #
        # Decision Tree Algorithm.
        # The default value of the maximum tree depth is 5, and the minimum number of instances in a node is 15.
        # max_tree_depth and minimum_instances are methods to control the trees in order for them not to grow too deep
        # and over-fit the data.
        #

        # Checks:
        # 1. Purity Check. If node is already pure, no need to proceed. We will just return the class of this pure node.
        # 2. If the # of instances in a node is less than the minimum # of instances allowed in a node, terminate.
        # 3. If we have already reached the maximum depth of the tree, terminate.
        if (self.purity_check(data) or len(data) < min_instances or depth_count == max_tree_depth):
            classified = self.class_label(data)
            return classified

        # we start the recursion of going down the sub-branches of the tree.
        else:
            depth_count += 1

            # We get a random subset of attributes that we wish to use for this tree with get_attributes
            attributes = self.get_attributes(data, number_features)
            # We get the best attribute and attribute values to split the instances.
            split_attribute, split_decision_value = self.get_best_split(data, attributes)
            # We pass the attribute that we decide to use for splitting, and the corresponding values used for
            # splitting the instances into the following function that does the splitting of the dataset for us.
            left_node_data, right_node_data = self.get_split(data, split_attribute, split_decision_value)

            # checking for empty nodes
            if len(left_node_data) == 0 or len(right_node_data) == 0:
                classified = self.class_label(data)
                return classified

            # Creating a dictionary of splits to remember how the algorithm splits the features
            split_criteria = '%s < %s' % (split_attribute, split_decision_value)
            sub_branches = {split_criteria: []}

            # Main recursion starts here.
            # Going down the tree - get the split from left_node_data and right_node_data and re-split it if possible.
            left = self.decision_trees(left_node_data, depth_count, min_instances,
                                       max_tree_depth, number_features)
            right = self.decision_trees(right_node_data, depth_count, min_instances,
                                        max_tree_depth, number_features)

            # If for a split, the resulting nodes have the same answers, then there's no point to create the split in
            # the first place. So our sub-branches in that case won't be a dictionary, instead it will just be one of
            # those two similar answers.
            if left == right:
                sub_branches = left

            else:
                sub_branches[split_criteria].append(left)
                sub_branches[split_criteria].append(right)

            return sub_branches

    def decision_tree_classification(self, instance, d_tree, counter=0):

        #
        # Function to make a prediction with the decision tree above.
        # d_tree, i.e the Decision Tree above returns a dict of a dict, containing recursive and embedded information
        # about the attributes we conducted the splits on. In order to understand each split, we have to recursively
        # "open" the dictionary.
        #

        # if decision tree immediately gives us a classification, which is not in the form of a dict object, then we
        # immediately return the value. Else we continue.
        if type(d_tree) != dict and counter == 0:
            return d_tree
        else:
            counter += 1
            split_criteria = list(d_tree.keys())[0]
            attribute_index, comparison, split_value = split_criteria.split(" ")

        # If our instance has the value of 0, i.e. less than 1, we pick the left node, else pick right.
        if instance[int(attribute_index)] < int(split_value):
            classification = d_tree[split_criteria][0]  # i.e. pick left node
        else:
            classification = d_tree[split_criteria][1]  # i.e. pick right node

        # If the returned variable is no longer a dictionary, it means that we have arrived at the leaf, and arrived
        # at a predicted class label. So we return the classified label.
        if not isinstance(classification, dict):
            return classification

        # if the returned variable is still a dict, it means there are still sub trees that we need to go down to reach
        # the leaf. We therefore recursively go down the remaining branches of the tree.
        else:
            remaining_sub_tree = classification
            return self.decision_tree_classification(instance, remaining_sub_tree)

    def bootstrapping(self, data, ratio):

        #
        # Bootstrapping procedure of random forest - Takes a dataset and samples the dataset with replacement to create
        # bootstrapped datasets.
        # ratio: proportion of the size of a bootstrapped dataset to the size of the entire complete training dataset.
        #

        random_sample_indices = np.random.randint(low=0, high=int(len(data)), size=int(round(ratio*len(data))))
        samples = data[random_sample_indices]
        # returns the subset of examples/instances.
        return samples

    def random_forest(self, data, n_decision_trees, ratio_bootstrap, min_instances, n_features, max_tree_depth,):

        #
        # Algorithm for Random Forest.
        #

        forest = []
        # n_decision_trees: number of decision trees used in this random forest to generate the vote at the end.
        for i in range(n_decision_trees):
            # Initialize and train all the different decision trees on different sets of the bootstrapped data
            # and on random subsets of attributes.
            bootstrapped_data = self.bootstrapping(data, ratio_bootstrap)
            d_trees = self.decision_trees(bootstrapped_data, 0, min_instances, max_tree_depth, n_features)
            forest.append(d_trees)
        return forest

    def random_forest_classification(self, instance, forest):

        #
        # Function to make a prediction using the trained random forest above.
        #

        # Dictionary containing the index of the random forests and their prediction for this particular instance.
        rf_classification = {}

        for i in range(len(forest)):
            classification = self.decision_tree_classification(instance, d_tree=forest[i], counter=0)
            rf_classification['tree %s' % i] = classification  # which tree made what prediction.

        # finding the mode amongst all predictions provided by each tree within the forest for this instance.
        action_value, count = np.unique(rf_classification.values(), return_counts=True)
        index = np.argmax(count)
        mode = action_value[index]
        random_forest_predictions = mode
        return random_forest_predictions

    ###########################################################################################################
    # ####################################################################################################### #
    # -------------------------------- Codes for Random Forests Ends Here ----------------------------------- #
    # ####################################################################################################### #
    ###########################################################################################################

    def final(self, state):
        # Tidy up when Pacman dies

        print "Game Ended!"
        print "Thank you for playing!"


    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)

        # Initializing a Random Forests and the constituent Decision Trees
        # Here, we are using 5 Decision Trees within this Random Forests, given that the training data is very limited.
        forest = self.random_forest(self.combined_data, 5, 0.3, 12, 8, 6)

        # Prediction generated by the random forest above.
        forest_prediction = self.random_forest_classification(features, forest)

        # Get the actions we can try.
        legal = api.legalActions(state)
        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.

        action = self.convertNumberToMove(forest_prediction)
        if action in legal:
            return api.makeMove(action, legal)
        else:
            return api.makeMove(random.choice(legal), legal)
