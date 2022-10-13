import numpy as np
import imutils
import cv2
import os
from imutils import paths
import math
import operator

def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++IMPORTING THE TRAINING DATA+++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

rawImages = []
features = []
labels= []
imagepaths = paths.list_images("C:/Users/Faizan/Downloads/ownloads/data/train")



for (i, imagePath) in enumerate(imagepaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)
    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)



rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++IMPORTING THE TESTING DATA++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


testImages = []
features1 = []
labels1 = []
test_image = paths.list_images("C:/Users/Faizan/Downloads/ownloads/data/test/cars")

for (j, imagePath1) in enumerate(test_image):
    image1 = cv2.imread(imagePath1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    cv2.GaussianBlur(image1, (3, 3), 0)
    label1 = imagePath1.split(os.path.sep)[-1].split(".")[0]
    pixels1 = image_to_feature_vector(image1)
    hist1 = extract_color_histogram(image1)
    testImages.append(pixels1)
    features1.append(hist1)
    labels1.append(label1)

# print(testImages)
testImages = np.array(testImages)
features1 = np.array(features1)
labels1 = np.array(labels1)


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] in predictions:
            correct = correct + 1
            print('Its a Car')
        else:
            print('Its a bike')
        dist = testSet[x][-1] - predictions
        print(dist)
    return (correct / float(len(testSet)) * 100)


def main():
    # prepare data
    trainingSet = []
    testSet = []
    predictions = []
    k = 2
    for x in range(len(testImages)):
        neighbors = getNeighbors(rawImages, testImages[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testImages[x][-1]))
    accuracy = getAccuracy(testImages, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()