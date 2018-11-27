import pickle
from visualization.visualize import visualize_images
import sys
import PPR
import operator
import datetime

fileName = sys.argv[1]
method = sys.argv[2] # KNN/PPR/ALL

DEBUG = 1

if method == "KNN":
	if len(sys.argv) < 4:
		K = 5
	else:
		K = int(sys.argv[3])
elif method == "PPR":
	graphPickle = sys.argv[3]
elif method == "ALL":
	K = int(sys.argv[3])
	graphPickle = sys.argv[4]

#graphPickle = "graph-k-10-20181124-155609.pkl"

if DEBUG:
	print("* DEBUG Mode On *")

startTime = datetime.datetime.now()
print("Start Time:", startTime)

'''
Runs the KNN Algorithm
'''
def KNN():
	print("KNN based classifier:")

	givenClasses = {}

	lineList = []
	with open(fileName) as fin:
	    for line in fin:
	    	lineList.append(line)
		
	for line in range(len(lineList)):
		if line < 2:
			continue
		b =  [x.strip() for x in lineList[line].strip().split(" ")]
		if b[0] not in givenClasses:
			givenClasses[int(b[0])] = []
		givenClasses[int(b[0])].append(b[-1])

	distanceMatrix = []
	with open('total_diffs.pkl', 'rb') as handle:
	    distanceMatrix = pickle.load(handle)

	if DEBUG:
		print("\tShape of Distance Matrix (with repeats):", distanceMatrix.shape)

	indexMatrix = []
	with open('images_list.pkl', 'rb') as handle:
	    indexMatrix = pickle.load(handle)

	if DEBUG:
		print("\tLength of Image - Index List: ", len(indexMatrix))

	classes = {}
	notIn = 0
	In = 0
	InCorrectIn = 0
	for imageId in indexMatrix:
		if imageId not in givenClasses:
			In += 1
			tempDistance = {}
			for givenImage in givenClasses:
				tempDistance[givenImage] = distanceMatrix[indexMatrix.index(givenImage)][indexMatrix.index(imageId)]
			sortedDistance = sorted(tempDistance.items(), key=lambda x: x[1])
			thisImageClass = {}
			for dis in sortedDistance[0:K]:
				classOfImage = givenClasses[dis[0]]
				for imgClass in classOfImage:
					if imgClass not in thisImageClass:
						thisImageClass[imgClass] = 0
					thisImageClass[imgClass] = thisImageClass[imgClass] + 1
			sortedClass = sorted(thisImageClass.items(), key=lambda x: x[1])
			# Add for multiple classes having same number of classes
			if imageId in classes:
				InCorrectIn += 1
			else:
				classes[imageId] = []
			tempK = 1
			for aa in range(len(sortedClass)-1):
				if(sortedClass[len(sortedClass)-1-aa][1] == sortedClass[len(sortedClass)-2-aa][1]):
					tempK += 1
			for aa in range(tempK):
				if sortedClass[((-1) - aa)][0] not in classes[imageId]:
					classes[imageId].append(sortedClass[((-1) - aa)][0])
		else:
			notIn += 1

	if DEBUG:
		print("\tImages not in the given images:", len(classes))
		print("\tGiven Images:", len(givenClasses))
		print("\tAssert This is given image:", notIn) # should be equal to givenClasses
		print("\tUnique ImageId not in given images + repeating:", In) # Should give the unique number of imageId
		print("\tImages that are repeating: ", InCorrectIn) # Should give repeated ImageId

	classDict = {}
	for j in classes:
		for lent in range(len(classes[j])):
			if classes[j][lent] not in classDict:
				classDict[classes[j][lent]] = []
			classDict[classes[j][lent]].append(j)

	for j in givenClasses:
		for lent in range(len(givenClasses[j])):
			if givenClasses[j][lent] not in classDict:
				classDict[givenClasses[j][lent]] = []
			classDict[givenClasses[j][lent]].append(j)

	if DEBUG:
		print("\tNumber of Classes:", len(classDict))

	for jj in classDict:
		visualize_images(jj, classDict[jj])

'''
	Sort the Dictionary by values and return smallest key
'''
def giveSmallest(givenDict):
	classifiedClasses = []
	sorted_d = sorted(givenDict.items(), key=operator.itemgetter(1),reverse=True)
	biggestVal = ""
	for tuples in sorted_d:
		if len(classifiedClasses) == 0:
			classifiedClasses.append(tuples[0])
			biggestVal = tuples[1]
		elif tuples[1] == biggestVal:
			classifiedClasses.append(tuples[0])
	return classifiedClasses

'''
	Runs the PPR based classification
'''
def PPRalgo():
	print("PPR based classifier:")
	givenClasses = {}

	lineList = []
	with open(fileName) as fin:
	    for line in fin:
	    	lineList.append(line)
		
	for line in range(len(lineList)):
		if line < 2:
			continue
		b =  [x.strip() for x in lineList[line].strip().split(" ")]
		if b[0] not in givenClasses:
			givenClasses[str(b[0])] = []
		if b[-1] not in givenClasses[str(b[0])]:
			givenClasses[str(b[0])].append(b[-1])

	if DEBUG:
		print(givenClasses)

	givenClassToImageId = {}

	for imageId in givenClasses:
		for classList in givenClasses[imageId]:
			if classList not in givenClassToImageId:
				givenClassToImageId[classList] = []
			if imageId not in givenClassToImageId[classList]:
				givenClassToImageId[classList].append(imageId)

	if DEBUG:
		print("Class to Image ID")
		print(givenClassToImageId)

	classRank = {}

	for classes in givenClassToImageId:
		inputStringList = ""
		for images in givenClassToImageId[classes]:
			inputStringList = inputStringList + images + ","
		inputStringList = inputStringList[:-1]
		if DEBUG:
			print(classes + " : " + inputStringList)

		classRank[classes] = PPR.list_PPR(inputStringList, graphPickle)
		if DEBUG:
			print(len(classRank[classes]))

	testClassToImageId = {}

	for classes in givenClassToImageId:
		testClassToImageId[classes] = []

	testClassifier = {}

	for classes in classRank:
		rankingList = classRank[classes]
		for imageId in rankingList:
			if imageId not in testClassifier:
				testClassifier[imageId] = {}
			testClassifier[imageId][classes] = rankingList[imageId]

	if DEBUG:
		print(len(testClassifier))

	testImageToClass = {}

	for imageId in testClassifier:
		testImageToClass[imageId] = giveSmallest(testClassifier[imageId])

	if DEBUG:
		for imageId in givenClasses:
			print(str(imageId) + ":")
			print("\tOriginal Classes:",  givenClasses[imageId])
			print("\tNew Classes:", testImageToClass[imageId])
			print("\tData:", testClassifier[imageId])
			print("\n\n")

	newTestDataClassified = givenClassToImageId

	for imageId in testImageToClass:
		if imageId not in givenClasses:
			for className in testImageToClass[imageId]:
				newTestDataClassified[className].append(imageId)


	for classes in newTestDataClassified:
		intVals = [int(x) for x in newTestDataClassified[classes]]
		visualize_images(classes, intVals)


# MAIN FUNCTION BEGINS HERE

if method == "KNN":
	KNN()
elif method == "PPR":
	PPRalgo()
elif method == "ALL":
	KNN()
	print("")
	PPRalgo()
else:
	print("Available Options: KNN, PPR, ALL")

endTime = datetime.datetime.now()
print("End Time: ", endTime)

print("Total Time: ", endTime - startTime)