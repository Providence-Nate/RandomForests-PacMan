# Random Forests Pac-Man
Design and Implementation of Pac-Man Strategies with random forests algorithm from scratch in a deterministic, fully observable Pacman Environment. Although not a very effective algorithm for game agent decision making, this project renders the opportunity to understanding of the basics of decision trees, their statistical aspects of decision making, and ensemble learning that creates the Random Forest algorithm.

# Design and Execution 
The entire decision-tree and random forests algorithm is coded from scratch, without any dependency on Scikit-Learn libraries, and hence provides users with the fundamental basics of how a decision tree algorithm truly functions. The decision-tree algorithm is trained with good-moves.txt which includes some data collected from codes that played Pacman, and some codes that actually won games of Pacman. Each line in good-moves.txt contains a feature vector, and a final digit - which encodes the action that Pacman should take. This final digit in the feature vector in good-moves.txt is what the classifier is trying to learn to classify.

# How to run
This version of Pac-Man requires the Python 2.7 environment. Run the following command to play the game:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python pacman.py --pacman ClassifierAgent --layout mediumClassic --n 20
