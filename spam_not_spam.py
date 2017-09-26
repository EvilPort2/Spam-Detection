import nltk
import random
import os
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
import string
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if os.name == 'nt':
	clear_screen = "cls"
else:
	clear_screen = "clear"
os.system(clear_screen)


def find_feature(word_features, message):
	# find features of a message
	feature = {}
	for word in word_features:
		feature[word] = word in message.lower()
	return feature

def create_mnb_classifier(trainingset, testingset):
    # Multinomial Naive Bayes Classifier
    print("\nMultinomial Naive Bayes classifier is being trained and created...")
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(trainingset)
    accuracy = nltk.classify.accuracy(MNB_classifier, testingset)*100
    print("MultinomialNB Classifier accuracy = " + str(accuracy))
    return MNB_classifier


def create_bnb_classifier(trainingset, testingset):
    # Bernoulli Naive Bayes Classifier
    print("\nBernoulli Naive Bayes classifier is being trained and created...")
    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(trainingset)
    accuracy = nltk.classify.accuracy(BNB_classifier, testingset)*100
    print("BernoulliNB accuracy percent = " + str(accuracy))
    return BNB_classifier


def create_logistic_regression_classifier(trainingset, testingset):
    # Logistic Regression classifier
    print("\nLogistic Regression classifier is being trained and created...")
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(trainingset)
    print("Logistic Regression classifier accuracy = "+ str((nltk.classify.accuracy(LogisticRegression_classifier, testingset))*100))
    return LogisticRegression_classifier


def create_sgd_classifier(trainingset, testingset):
    print("\nSGD classifier is being trained and created...")
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(trainingset)
    print("SGD Classifier classifier accuracy = " + str((nltk.classify.accuracy(SGDClassifier_classifier, testingset))*100))
    return SGDClassifier_classifier

def create_nb_classifier(trainingset, testingset):
    # Naive Bayes Classifier
    print("\nNaive Bayes classifier is being trained and created...")
    NB_classifier = nltk.NaiveBayesClassifier.train(trainingset)
    accuracy = nltk.classify.accuracy(NB_classifier, testingset)*100
    print("Naive Bayes Classifier accuracy = " + str(accuracy))
    NB_classifier.show_most_informative_features(20)
    return NB_classifier

def create_training_testing():
	"""
	function that creates the feature set, training set, and testing set
	"""
	with open('SMSSpamCollection') as f:
		messages = f.read().split('\n')

	print("Creating bag of words....")
	all_messages = []														# stores all the messages along with their classification
	all_words = []															# bag of words
	for message in messages:			
		if message.split('\t')[0] == "spam":
			all_messages.append([message.split('\t')[1], "spam"])
		else:
			all_messages.append([message.split('\t')[1], "ham"])
		
		for s in string.punctuation:										# Remove punctuations
			if s in message:
				message = message.replace(s, " ")
		
		stop = stopwords.words('english')
		for word in message.split(" "):										# Remove stopwords
			if not word in stop:
				all_words.append(word.lower())
	print("Bag of words created.")

	random.shuffle(all_messages)
	random.shuffle(all_messages)
	random.shuffle(all_messages)

	all_words = nltk.FreqDist(all_words)
	word_features = list(all_words.keys())[:2000]							# top 2000 words are our features

	print("\nCreating feature set....")
	featureset = [(find_feature(word_features, message), category) for (message, category) in all_messages]
	print("Feature set created.")
	trainingset = featureset[:int(len(featureset)*3/4)]
	testingset = featureset[int(len(featureset)*3/4):]

	print("\nLength of feature set ", len(featureset))
	print("Length of training set ", len(trainingset))
	print("Length of testing set ", len(testingset))

	return word_features, featureset, trainingset, testingset

def main():
	"""
	this function is used to show how to use this program.
	the models can be pickled if wanted or needed.
	i have used 4 mails to check if my models are working correctly.
	"""

	word_features, featureset, trainingset, testingset = create_training_testing()
	NB_classifier = create_nb_classifier(trainingset, testingset)
	MNB_classifier = create_mnb_classifier(trainingset, testingset)
	BNB_classifier = create_bnb_classifier(trainingset, testingset)
	LR_classifier = create_logistic_regression_classifier(trainingset, testingset)
	SGD_classifier = create_sgd_classifier(trainingset, testingset)
	"""DT_classifier = create_dt_classifier(trainingset, testingset)"""

	mails = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",\
	 "Hello Ward, It has been almost 3 months since i have written you. Hope you are well.", \
	 "FREE FREE FREE Get a chance to win 10000 $ for free. Also get a chance to win a car and a house",\
	 "Hello my friend, How are you? It is has been 3 months since we talked. Hope you are well. Can we meet at my place?"]
	
	print("\n")
	print("Naive Bayes")
	print("-----------")
	for mail in mails:
		feature = find_feature(word_features, mail)
		print(NB_classifier.classify(feature))
		
	print("\n")
	print("Multinomial Naive Bayes")
	print("-----------")
	for mail in mails:
		feature = find_feature(word_features, mail)
		print(MNB_classifier.classify(feature))
		
	print("\n")
	print("Bernoulli Naive Bayes")
	print("-----------")
	for mail in mails:
		feature = find_feature(word_features, mail)
		print(BNB_classifier.classify(feature))
		
	print("\n")
	print("Logistic Regression")
	print("-----------")
	for mail in mails:
		feature = find_feature(word_features, mail)
		print(LR_classifier.classify(feature))
		
	print("\n")
	print("Stochastic Gradient Descent")
	print("-----------")
	for mail in mails:
		feature = find_feature(word_features, mail)
		print(SGD_classifier.classify(feature))
		 
		
main()