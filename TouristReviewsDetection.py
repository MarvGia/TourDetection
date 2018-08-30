import json
import time
import math
import random
from math import radians, cos, sin, asin, sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model

 
#import sklearn
#from sklearn import svm
#from sklearn import liblinearutil
#from sklearn import liblinear
#from svmutil import *



import numpy as np 
import pandas as pd
from sklearn.cluster import DBSCAN

#Dictionaries
textType = {}
stringsIds = {}
idsString = {}
reviewsUserIdBusinessIdText = {}
reviewsUserIdBusinessIdDate = {}
reviewsUserIdBusinessId = {}
reviewsUserIdText = {}
reviewsCountReviews = {}
users = {}
BusinessIdLongitude = {} #x
BusinessIdLatitude = {} #y
kNearestNeighbors = {}
latLonNameBusinessId = {}
latLonName = {}
myList = []
markedBusinesses = []
reviewsTextBusinessId = {}

counter50=0
counter100=0
counter200=0
counter300=0
counter400=0
counter0 = 0

#counters
stringIdCounter = 0
averageClustersPerUser = 0
usersCounter = 0
businessCounter = 0
activeUsersCounter = 0

#LOF vars
maxKm = 200
counter = 0 
check = False
k = 7 #neighbors

minimumNumberOfReviewsPerUser = 50 

#DBSCAN vars
kms_per_radian = 6371.0088
epsilon = 200 / kms_per_radian
min_samples = 35

standardOutliers = 0
totalPointsChecked = 0
extraOutliers = 0

#Sampling Dictionaries
tourReviews = {}
tourReviewsCounter = 0
localReviews = {}
localReviewsCounter = 0
torontoTourReviews = {}
torontoLocalReviews = {}


sampleSize = 645 #gia ola sampleSize = 176004
#sampleSize = 2184 
#sampleSize = 11160 #analogia 1:1
#sampleSize = 16740 #66 analogia 1:2
#sampleSize = 22320 #analogia 1:3
tourItems = []
localItems = []
        

def findActualTourReviews():
    global myList
    f = open('tourReviews.txt','r',encoding = 'utf8')
    line = f.readline()
    #line = line.strip('\n')
    
    text = ''
    while(line!=''):
        if '{' in line:
            #if (len(line)==2):
                #line = f.readline()
                #continue
            while( '}' not in line):
                
                line = f.readline()
                if '}' not in line:
                    text = text + line
                
            myList.append(text.strip())
            #f.write('{\n'+text+'}\n')
        line = f.readline()
        text = ''
    #print (str(len(myList)))
def findDistance(lat1,lng1,lat2,lng2):
    AVG_EARTH_RADIUS = 6371.0088  # in km

    """ Calculate the great-circle distance bewteen two points on the Earth surface.

    :input: two 2-tuples, containing the latitude and longitude of each point
    in decimal degrees.

    Example: findDistance(45.7597, 4.8422, 48.8567, 2.3508)

    :output: Returns the distance in kilometers between the two points.

    """

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    # calculate distance
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
    
    return h  # in kilometers


def parseReviewFile():
    global activeUsersCounter,reviewsUserIdBusinessIdText,reviewsTextBusinessId
    print ('Parsing review file...')
    totalReviews = 0
    reviewsTested = 0
    businessList =[]
    textList = []
    tempList = []
    global users

    with open('yelp_academic_dataset_review.json',encoding="utf8") as reviewsFile: 
        f = reviewsFile.readline()
        while(f!=''):      
            json_data=f
            data = json.loads(json_data)
            if data['user_id'] in users:
                if data['user_id'] in reviewsUserIdBusinessId:
                    reviewsCountReviews[data['user_id']] = reviewsCountReviews.get(data['user_id']) + 1
                    tempList = reviewsUserIdBusinessId.get(data['user_id'])[:]
                    tempList.append(data['business_id'])
                    reviewsUserIdBusinessId[data['user_id']] = tempList
                    
                    tempList = reviewsUserIdText.get(data['user_id'])[:]
                    tempList.append(data['text'])
                    reviewsUserIdText[data['user_id']] = tempList
                    reviewsTextBusinessId[data['text']] = data['business_id']
                    reviewsUserIdBusinessIdText[data['user_id'],data['business_id']] = data['text']
                    reviewsUserIdBusinessIdDate[data['user_id'],data['business_id']] = data['date']
                else:
                    reviewsCountReviews[data['user_id']] = 1
                    businessList.append(data['business_id'])
                    reviewsUserIdBusinessId[data['user_id']] = businessList[:]
                    
                    textList.append(data['text'])
                    reviewsUserIdText[data['user_id']] = textList[:]
                    reviewsTextBusinessId[data['text']] = data['business_id']
                    reviewsUserIdBusinessIdText[data['user_id'],data['business_id']] = data['text']
                    reviewsUserIdBusinessIdDate[data['user_id'],data['business_id']] = data['date']
                    del businessList[:]
                    del textList[:]
                
            totalReviews = totalReviews + 1
            f = reviewsFile.readline()
        
        for key in reviewsCountReviews:
            #print ('megethos reviewsCountReviews'+str(len(reviewsCountReviews)))
            if (reviewsCountReviews[key] >=minimumNumberOfReviewsPerUser):
                activeUsersCounter = activeUsersCounter + 1
                reviewsTested = reviewsTested + reviewsCountReviews[key]
            else:
                users.pop(key,None)
              
        print ('The number of reviews is: '+str(totalReviews)+'.\t'+'The number of reviews taking into account is: '+str(reviewsTested)+'.')
        print ('The number of users is: '+str(usersCounter)+".\t"+'The number of users taking into account is: '+str(activeUsersCounter)+'.')
        print ('The number of businesses is: '+str(businessCounter))
           
def parseBusinessFile():
    global businessCounter,latLonNameBusinessId,markedBusinesses
    tempList = []
    tempList1 = []
    categories = []
    print ('Parsing business file...')
    with open('yelp_academic_dataset_business.json',encoding="utf8") as businessFile:
        f = businessFile.readline()
        while(f!=''):    
            json_data = f  
            data = json.loads(json_data)
            if (data['city'] == 'Toronto'):
                categories = data['categories']
                if categories is not None:
                    for i in range (len(categories)):
                        if (categories[i] == 'Restaurants' or categories[i] == 'Bars' or categories[i] == 'Cafes'):
                            markedBusinesses.append(data['business_id'])
                #tokens = categories.split(',')
                #print ('categories: '+str(categories[0]))
                
                        
                
            BusinessIdLongitude[data['business_id']] = data['longitude']
            BusinessIdLatitude[data['business_id']] = data['latitude']
            tuple1 = (data['longitude'],data['latitude'])
            if tuple1 in latLonName.keys():
                tempList = latLonName.get((data['longitude'],data['latitude']))[:]
                tempList.append(data['name'])
                latLonName[(data['longitude'],data['latitude'])] = tempList[:]
            else:
                tempList.append(data['name'])
                latLonName[(data['longitude'],data['latitude'])] = tempList[:]
            tuple1 = (data['longitude'],data['latitude'],data['name'])
            if tuple1 in latLonNameBusinessId:
                tempList1 = latLonNameBusinessId.get((data['longitude'],data['latitude'],data['name']))[:]
                tempList1.append(data['business_id'])
                latLonNameBusinessId[(data['longitude'],data['latitude'],data['name'])] = tempList1 [:]
            else :
                tempList1.append(data['business_id'])
                latLonNameBusinessId[(data['longitude'],data['latitude'],data['name'])] = tempList1[:]
            
            del tempList[:]
            del tempList1[:]
            f = businessFile.readline()
            businessCounter = businessCounter + 1
def writeTourReviewsOnfile (key,outliers,f1): #isos prepei na aferethi to key
    global latLonName,latLonNameBusinessId,reviewsUserIdBusinessIdText
    for i in range (0,len(outliers)):
            secondBreak = False
            tempList= latLonName.get((outliers[i][0],outliers[i][1]))[:]
            for j in range(0,len(tempList)):
                tempList1 = latLonNameBusinessId.get((outliers[i][0],outliers[i][1],tempList[j]))[:]
                for k in range(0,len(tempList1)):
                    tempBusinessId = tempList1[k]
                    if reviewsUserIdBusinessIdText.get((key,tempBusinessId)) is not None:
                        secondBreak = True
                        text = reviewsUserIdBusinessIdText.get((key,tempBusinessId))
                        if text is None: #prepei na aferethoun
                            print ('after if else '+str(key)+' , '+str(tempBusinessId)) #prepei na aferethoun
                        f1.write('review #'+str(i+1)+'\t '+reviewsUserIdBusinessIdDate.get((key,tempBusinessId))+'\n{\n'+str(text)+'\n}\n')
                        #genika kai oxi gia 100 
                        #f1.write('review #'+str(i+1)+'\t '+reviewsUserIdBusinessIdDate.get((key,tempBusinessId))+'\n'+str(text)+'\n')
                        break
                if secondBreak == True:
                    break
def addReviewToGroundTruth(key,cluster,reviewType):
    global textType,tourReviews,tourReviewsCounter,localReviews,localReviewsCounter,latLonName,latLonNameBusinessId,reviewsUserIdBusinessIdText
    for i in range (0,len(cluster)):
        secondBreak = False
        tempList= latLonName.get((cluster[i][0],cluster[i][1]))[:]
        for j in range(0,len(tempList)):
            tempList1 = latLonNameBusinessId.get((cluster[i][0],cluster[i][1],tempList[j]))[:]
            for k in range(0,len(tempList1)):
                tempBusinessId = tempList1[k]
                if reviewsUserIdBusinessIdText.get((key,tempBusinessId)) is not None:
                    secondBreak = True
                    text = reviewsUserIdBusinessIdText.get((key,tempBusinessId))
                    if (reviewType == 'tourism'):
                        textType[text] = 1
                        tourReviews[tourReviewsCounter] = text
                        tourReviewsCounter = tourReviewsCounter + 1
                    else :
                        localReviews[localReviewsCounter] = text
                        localReviewsCounter = localReviewsCounter + 1
                        textType[text] = 0
                    break
            if secondBreak == True:
                break            
def initItems(tourReviews,localReviews,analogia):
    global tourItems,localItems
    
    if (analogia == 1):
        for i in range (0,len(tourReviews)):
            tourItems.append(i)
            
        for i in range (0,len(localReviews)):
            localItems.append(i)
    else:
        for i in range (0,len(tourReviews)):
            tourItems.append(i)
        #j = len(tourReviews)
        for i in range (0,len(localReviews)):
            j = len(tourReviews) + i
            localItems.append(j)
def initItems8020(tourReviews,localReviews,analogia):
    global tourItems,localItems
    dictCounter = 0
    dictCounter1 = 0
    tourReviewsCounter8020 = 0
    localReviewsCounter8020 = 0
    tourReviewsCounter80201 = 0
    localReviewsCounter80201 = 0
    counterTourReviews = 0
    counterLocalReviews = 0
    newDict = {}
    newDict1 = {}
    chosenItems80Temp = []
    balancedChosenItems80 = []
    
    
    for i in range (0,len(tourReviews)):
            tourItems.append(i)
     
    for i in range (0,len(localReviews)):
            j = len(tourReviews)+i
            localItems.append(j)
           
    if (analogia == 0):
        tourAndLocalItems = localItems+tourItems
        
        a = math.floor(len(tourAndLocalItems) * 0.8)
        
        chosenItems80 = random.sample(tourAndLocalItems,  a)
        for i in range(0,len(chosenItems80)):
            tourAndLocalItems.remove(chosenItems80[i])
            
        chosenItems20 = tourAndLocalItems
        for i in range(0,len(chosenItems80)):
            if (chosenItems80[i]<len(tourItems)):
                balancedChosenItems80.append(chosenItems80[i])
                counterTourReviews = counterTourReviews + 1
        for i in range (0,len(chosenItems80)):
            if (chosenItems80[i]>= len(tourItems)):
                counterLocalReviews = counterLocalReviews + 1
                balancedChosenItems80.append(chosenItems80[i])
                if (counterLocalReviews>=counterTourReviews):
                    break
        print ('arithmos apo touristika sto 80%: '+str(counterTourReviews))
        print ('arithmos apo topika sto 80%: '+str(counterLocalReviews))
        print ('megethos deigmatos 80%: '+str(len(balancedChosenItems80)))
    '''    
    if (analogia == 1): #afti dn xriazete pithanotata
        a = len(localReviews) - len(tourReviews)/2
        b = len(tourReviews)/2
        chosenItems80 = random.sample(localItems, int(a))
        for i in range (len(chosenItems80)):
            localItems.remove(chosenItems80[i])
        chosenItems80Temp = random.sample(tourItems, int(b))
        for i in range (len(chosenItems80Temp)):
            tourItems.remove(chosenItems80Temp[i])
        chosenItems80 = chosenItems80 + chosenItems80Temp
        
        chosenItems20 = random.sample(localItems, int(len(tourReviews)/2))
        chosenItems20 = chosenItems20 + random.sample(tourItems, int(len(tourReviews)/2))
    '''   
    #print ("80 : "+str(len(chosenItems80))+', 20: '+str(len(chosenItems20)))
    ''    
    for i in range(0,len(balancedChosenItems80)): 
        if (balancedChosenItems80[i]<len(tourReviews)):
            newDict[dictCounter] = tourReviews.get(balancedChosenItems80[i])
            dictCounter = dictCounter + 1 
            tourReviewsCounter8020 = tourReviewsCounter8020 + 1
        else:
            newDict[dictCounter] = localReviews.get(balancedChosenItems80[i]-len(tourReviews))
            dictCounter = dictCounter + 1 
            localReviewsCounter8020 = localReviewsCounter8020 + 1
    print ('localReviewsCounter8020: '+str(localReviewsCounter8020)+ ' tourReviewsCounter8020: '+str(tourReviewsCounter8020))
    for i in range(0,len(chosenItems20)): 
        if (chosenItems20[i]<len(tourReviews)):
            newDict1[dictCounter1] = tourReviews.get(chosenItems20[i])
            dictCounter1 = dictCounter1 + 1 
            tourReviewsCounter80201 = tourReviewsCounter80201 + 1
        else:
            newDict1[dictCounter1] = localReviews.get(chosenItems20[i]-len(tourReviews))
            dictCounter1 = dictCounter1 + 1   
            localReviewsCounter80201 = localReviewsCounter80201 + 1
    print ('localReviewsCounter80201: '+str(localReviewsCounter80201)+ ' tourReviewsCounter80201: '+str(tourReviewsCounter80201))  
    
    return (newDict,newDict1)
def torontoReviews():
    global tourReviews,localReviews,reviewsTextBusinessId,markedBusinesses,torontoTourReviews,torontoLocalReviews
    
    counter = 0
    counter1 = 0
    for i in range(0,len(tourReviews)):
        businessId = reviewsTextBusinessId.get(tourReviews[i])
        if businessId in markedBusinesses:
            torontoTourReviews[counter] = tourReviews[i]
            counter = counter + 1
            
    for i in range(0,len(localReviews)):
        businessId = reviewsTextBusinessId.get(localReviews[i])
        if businessId in markedBusinesses:
            torontoLocalReviews[counter1] = localReviews[i]
            counter1 = counter1 + 1
    #print (str(len (torontoLocalReviews))+' '+str(len(torontoTourReviews)))
def createSample(tourReviews,localReviews,analogia):
    global tourItems,localItems
    dictCounter = 0
    
    newDict = {}
    chosenTourItems = []
    chosenLocalItems = []
    chosenTourLocalItems = []
    chosenItems = []
    
   
    #analogia tixea
    if (analogia == 0):
        chosenTourLocalItems = tourItems + localItems
        chosenItems = random.sample(chosenTourLocalItems,  sampleSize)
        #print (str(chosenItems))
        for i in range(0,len(chosenItems)):
            if (chosenItems[i] < len(tourReviews)):
                tourItems.remove(chosenItems[i])
            else:
                localItems.remove(chosenItems[i])
    #analogia touristikon 1:1
    if (analogia == 1):
        chosenTourItems = random.sample(tourItems,  int(sampleSize/2))
        for i in range(0,len(chosenTourItems)):
            tourItems.remove(chosenTourItems[i])
            
            
        chosenLocalItems = random.sample(localItems,  int(sampleSize/2))
        for i in range(0,len(chosenLocalItems)):
            localItems.remove(chosenLocalItems[i])
    
    #analogia touristikon 1:2 
    if (analogia == 2):
        chosenTourItems = random.sample(tourItems,  int(sampleSize/3))
        for i in range(0,len(chosenTourItems)):
            tourItems.remove(chosenTourItems[i])
        
        
        chosenLocalItems = random.sample(localItems,  int(2*(sampleSize/3)))
        for i in range(0,len(chosenLocalItems)):
            localItems.remove(chosenLocalItems[i])
    #analogia 1:3
    if (analogia == 3):
        chosenTourItems = random.sample(tourItems,  int(sampleSize/4))
        for i in range(0,len(chosenTourItems)):
            tourItems.remove(chosenTourItems[i])
            
            
        chosenLocalItems = random.sample(localItems,  int(3*(sampleSize/4)))
        for i in range(0,len(chosenLocalItems)):
            localItems.remove(chosenLocalItems[i])
           
    if (analogia == 1 or analogia == 2 or analogia == 3):
        for i in range(0,len(chosenTourItems)):    
            newDict[dictCounter] = tourReviews.get(chosenTourItems[i])
            dictCounter = dictCounter + 1
           
        for i in range(0,len(chosenLocalItems)):    
            newDict[dictCounter] = localReviews.get(chosenLocalItems[i])
            dictCounter = dictCounter + 1 
    else:
        for i in range (0,(len(chosenItems))):
            if (chosenItems[i]<len(tourReviews)):
                newDict[dictCounter] = tourReviews.get(chosenItems[i])
                dictCounter = dictCounter + 1 
            else:
                newDict[dictCounter] = localReviews.get(chosenItems[i]-len(tourReviews))
                dictCounter = dictCounter + 1 
    return newDict
def showPointsOfUsers(userId,f): 
         
    tempList = []
    tempList = reviewsUserIdBusinessId.get(userId)[:]
    for i in range(0,len(tempList)):
        coordinateX= str(BusinessIdLongitude.get(tempList[i]))
        coordinateY= str(BusinessIdLatitude.get(tempList[i]))
        f.write ('\nX: '+coordinateX+'\t'+'Y:'+coordinateY)
        
def findKNearest(lon,lat,points,index,k):
    global maxKm
    distances = []
    tempDistances = []
    neighbors = []
    global check 
    
    check = False  
    for i in range(0,len(points)):
        if (findDistance(lon,lat,BusinessIdLongitude.get(points[i]),BusinessIdLatitude.get(points[i])) == 0):
            tempDistances.append(1000000.0)
        else:
            tempDistances.append(findDistance(lon,lat,BusinessIdLongitude.get(points[i]),BusinessIdLatitude.get(points[i])))
    distances = tempDistances[:]
    #print ('pinakas apostasewn simion print to sortarisma: '+str(tempDistances))
    distances.sort()
    #print ('pinakas apostasewn simion: '+str(distances))
    if (distances[len(distances)-2] >= maxKm):
        check = True
        
    for i in range (0,k):
        neighbors.append(tempDistances.index(distances[i]))
    kNearestNeighbors[index]=neighbors
     
    return (distances) 
def density(lon,lat,user_id,i,k):
    distances = []
    sum1 = 0
    distances = findKNearest(lon,lat,reviewsUserIdBusinessId.get(user_id)[:],i,k)
    for i in range(0,k):
        sum1 = sum1 + distances[i]
#print ('sum apostasewn: '+str(sum1))    
    density = (sum1/k) 
    #print ('density: '+str(density)) 
    return (density)
def LOF(key,f):
    global check,counter
    densities = []
    densitiesCheck = []
    meanRelativeDensities = []
    sum1 = 0
    kNearestNeighborsList = []
    points = []
    
    counter = counter + 1
    
    f.write('\nUserId: '+key)
    
    showPointsOfUsers(key,f)
    points = reviewsUserIdBusinessId.get(key)[:]
    
    k = math.ceil((len(points) * 0.3))
    
    for i in range (0,len(points)):
        densities.append(density(BusinessIdLongitude.get(points[i]),BusinessIdLatitude.get(points[i]),key,i,k)) 
        #print ('check value: '+str(check))  
        densitiesCheck.append(check)
      
    
    
           
    for i in range (0,len(points)):
        if densitiesCheck[i] == True:
            for j in range (0,k):
                kNearestNeighborsList = kNearestNeighbors.get(i)[:]
                sum1 = sum1 + densities[kNearestNeighborsList[j]]
            #print ('to sum ton mean densities: '+str(sum1))
            #print ('ta index ton kontinoteron gitonon einai: '+str(kNearestNeighbors.get(i)[:]))  
            #print ('to mean density einai: '+str(densities[i]/((sum1/k))))  
            meanRelativeDensities.append(densities[i]/((sum1/k)))
            sum1 = 0
        else:
            meanRelativeDensities.append(0.0)
    f.write('\n ta skor einai: '+str(meanRelativeDensities))
    del densities[:]
    del meanRelativeDensities[:]    
def parseUserFile():  
    global usersCounter
    
    print ('Parsing user file...')     
    with open('yelp_academic_dataset_user.json',encoding="utf8") as userFile:
        f = userFile.readline()
        while(f!=''):
            json_data = f  
            data = json.loads(json_data)
            if (data['review_count'] >= minimumNumberOfReviewsPerUser):
                users[data['user_id']]=' '
            f = userFile.readline()       
            usersCounter = usersCounter + 1
def runDBScanToSampleOf50Users(file,file1):
    with open('sampleOf50Users.txt', 'r') as sampleFile:
        key = sampleFile.readline()
        key = key.strip('\n')
        
        while(key!=''):
            dbScan(key,file,file1)
            key = sampleFile.readline()
            key = key.strip('\n')
    
    sampleFile.close()  
def runDBScanToAllUsers(file,file1):
    counter = 0
    for key in users:
        #print (key)
        dbScan(key,file,file1)
        #counter = counter + 1
        #if (counter>=50):
            #break
    
def writeOnFileDBScanDetails(file):
    file.write('Method of finding outliers: DBSCAN')
    file.write('\nProgram Parameters \n')
    file.write('epsilon: '+str(epsilon*6371.0088)+', '+'minimum number of samples per cluster: '+str(min_samples)+'\n\n')
def writeOnFileLOFDetails(file):
    file.write('Method of finding outliers: LOF Algorithm')
    file.write('Program Parameters \n')
    file.write('maximum distance to count as inlier: '+str(maxKm)+', '+'minimum number of reviews per user: '+str(minimumNumberOfReviewsPerUser)+'\n\n')             
def dbScan(key,f,f1):
    global counter0,counter400,counter300,counter50,counter100,counter200,tourReviews,localReviews,averageClustersPerUser,counter,epsilon,standardOutliers,totalPointsChecked,extraOutliers,upperBoundForClustering,latLonNameBusinessId,reviewsUserIdBusinessIdDate,reviewsUserIdBusinessIdText,latLonName
    points = []
    #counter = counter + 1

    
    f.write('\nUserId: '+key)
    f1.write('\nUserId: '+key)
    
    showPointsOfUsers(key,f)
    points = reviewsUserIdBusinessId.get(key)[:]
    totalPointsChecked = totalPointsChecked + len(points)

    
    newcoords = np.zeros(shape=(len(points),2))
    for i in range(0,len(points)):
        newcoords[i]=[BusinessIdLongitude.get(points[i]),BusinessIdLatitude.get(points[i])]

    db = DBSCAN(eps=epsilon,min_samples=35, algorithm='ball_tree', metric='haversine').fit(np.radians(newcoords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([newcoords[cluster_labels == n] for n in range(num_clusters)])
    averageClustersPerUser = averageClustersPerUser + num_clusters
    f.write('\nNumber of clusters: {}\n'.format(num_clusters)) 
    outliers = newcoords[cluster_labels == -1]
    standardOutliers = standardOutliers + len(outliers)
    
    if (len(outliers!=0)):
        f1.write ('\nta outliers reviews einai : \n')
        writeTourReviewsOnfile(key,outliers, f1)    #prosthesame to key
    

    addReviewToGroundTruth(key,outliers,'tourism')
    
    for i in range (num_clusters):
        addReviewToGroundTruth(key,clusters[i],'local')
    '''
    for i in range (0,num_clusters):
        #if num_clusters == 1:
        if (len(clusters[i])!=0):
            for k in range(len(clusters[i])):
                
                for j in range(k+1,len(clusters[i])):
                    #j = k + 1    
                    if (findDistance(clusters[i][k][0],clusters[i][k][1],clusters[i][j][0],clusters[i][j][1])>400):
                        counter400 = counter400 + 1
                    elif (findDistance(clusters[i][k][0],clusters[i][k][1],clusters[i][j][0],clusters[i][j][1])>300):
                        counter300 = counter300 + 1
                    elif (findDistance(clusters[i][k][0],clusters[i][k][1],clusters[i][j][0],clusters[i][j][1])>200):
                        counter200 = counter200 + 1
                    elif (findDistance(clusters[i][k][0],clusters[i][k][1],clusters[i][j][0],clusters[i][j][1])>100):
                        counter100 = counter100 + 1
                    #else:#(findDistance(clusters[0][k][0],clusters[0][k][1],clusters[0][j][0],clusters[0][j][1])>50):
                        #counter50 = counter50 + 1
                    
                    
                    elif (findDistance(clusters[i][k][0],clusters[i][k][1],clusters[i][j][0],clusters[i][j][1])>50):
                        counter50 = counter50 + 1
                    else:#elif ((findDistance(clusters[i][k][0],clusters[i][k][1],clusters[i][j][0],clusters[i][j][1]))<100):
                        counter0 = counter0+1
                        
                        #print (findDistance(clusters[0][k][0],clusters[0][k][1],clusters[0][j][0],clusters[0][j][1]),' megaliteri apo 100 <--------')
                    
                        #print (findDistance(clusters[0][k][0],clusters[0][k][1],clusters[0][j][0],clusters[0][j][1]),' megaliteri apo 100 <--------')
                  
                    
                    
                    
                        #print (findDistance(clusters[0][k][0],clusters[0][k][1],clusters[0][j][0],clusters[0][j][1]),' megaliteri apo 100 <--------')
    '''
    f.write ('ta outliers einai: '+str(outliers))
def computeTFIDF(sample,f2,f3,f5):
    global textType,stringsIds,stringIdCounter,idsString
    corpus = []
    features = []
    newFeatures = []
    b = []
    c = []
    d = []
    tokens = []
    
    f2.write("text processing to sample started...")
    for i in range (0,len(sample)):
        corpus.append(sample.get(i))
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english', sublinear_tf=True)
    tfidf_matrix =  tf.fit_transform(corpus)
    #print (str(tfidf_matrix))
    feature_names = tf.get_feature_names()
    
    for i in range(0,len(sample)):     
        doc = i
        text = sample.get(i) 
        f2.write ('\ntext#'+ str(i+1) +' : \n'+text)
        feature_index = tfidf_matrix[doc,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
        
        f3.write(str(textType.get(text))+' ')
        #f5.write(str(textType.get(text))+'\n')
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            if w not in stringsIds:
                stringsIds[w] = stringIdCounter+1
                idsString[stringsIds.get(w)] = w  
                stringIdCounter = stringIdCounter + 1
            
            features.append(str(stringsIds.get(w))+':'+str(s))
            f2.write('\n')
            f2.write (str(w)+'\t'+str(s))
            
        for i in range (0,len(features)):
            tokens = features[i].split(':')
            b.append(int(tokens[0]))
            c.append(float(tokens[1]))
        d = sorted(range(len(b)), key=lambda k: b[k])
        b.sort()
        for i in range(0,len(b)):
            newFeatures.append(str(b[i])+':'+str(c[d[i]]))    
                
        for i in range (0,len(newFeatures)):
            if i!= len(newFeatures)-1:
                f3.write(newFeatures[i]+' ')
            else:
                f3.write(newFeatures[i])    
        f3.write('\n')    

        del tokens[:],b[:],c[:],d[:]
        del features[:]
        del newFeatures[:]
    f2.write("\ntext processing ended...")                                                         
if __name__ == '__main__': 
    startTimeOverall = time.time()
    parseUserFile()
    elapsedTime = time.time() - startTimeOverall
    print ('Time for parsing user file: '+'%.2f' %elapsedTime+' seconds.')
    startTime = time.time()
    parseBusinessFile()
    elapsedTime = time.time() - startTime 
    print ('Time for parsing business file: '+'%.2f' %elapsedTime+' seconds.')
    
    startTime = time.time()
    parseReviewFile()
    elapsedTime = time.time() - startTime
    print ('Time for parsing review file: '+'%.2f' %(elapsedTime/60)+' minutes.')
    
    f = open('resultsOfDBScan.txt', 'w')
    f1 = open ('outliersReviews.txt','w',encoding='utf8')
    f2 = open ('TFIDFResults.txt','a',encoding='utf8')
    f3a = open('Sample1.txt','w')
    f3b = open('Sample2.txt','w')
    f3c = open('Sample3.txt','w')
    f3d = open('Sample4.txt','w')
    f3e = open('Sample5.txt','w')
    f4 = open ('words.txt','w',encoding='utf8')
    f5a = open ('categorySample1.txt','w')
    f5b = open ('categorySample2.txt','w')
    f5c = open ('categorySample3.txt','w')
    f5d = open ('categorySample4.txt','w')
    f5e = open ('categorySample5.txt','w')
    
    '''
    #LOF   
    writeOnFileLOFDetails(f)
    '''
    
    #DBSCAN
    writeOnFileDBScanDetails(f)
    writeOnFileDBScanDetails(f1)
    
    #findActualTourReviews()
    #runDBScanToSampleOf50Users(f,f1)
    #print ('tourDictionaryLen '+str(len(tourReviews)))
    runDBScanToAllUsers(f, f1)
    
    
    
    
    '''
    #torontoReviews()
    #cross Validation
    initItems(torontoTourReviews,torontoLocalReviews,1) #prosoxi sinithos 1
    
    sample1 = createSample(torontoTourReviews,torontoLocalReviews,2)
    computeTFIDF(sample1, f2,f3a,f5a)
    
    sample2 = createSample(torontoTourReviews,torontoLocalReviews,2)
    computeTFIDF(sample2, f2,f3b,f5b)

    sample3 = createSample(torontoTourReviews,torontoLocalReviews,2)
    computeTFIDF(sample3, f2,f3c,f5c)
    
    sample4 = createSample(torontoTourReviews,torontoLocalReviews,2)
    computeTFIDF(sample4, f2,f3d,f5d)
    
    sample5 = createSample(torontoTourReviews,torontoLocalReviews,2)
    computeTFIDF(sample5, f2,f3e,f5e)
    
    print (str(len(sample1)))
    print (str(len(sample2)))
    print (str(len(sample3)))
    print (str(len(sample4)))
    print (str(len(sample5)))
    '''
    
    
    
    #80-20
    
    torontoReviews()
    
    print ('localToronto: '+str(len(torontoLocalReviews))+" tourToronto: "+str(len(torontoTourReviews))+" local: "+str(len(localReviews))+" tour: "+str(len(tourReviews)))
    
    print ("ksekinaei sampling")
    sample80,sample20 = initItems8020(torontoTourReviews,torontoLocalReviews,0)
    print("ksekinaei tfidf")
    computeTFIDF(sample80, f2, f3a, f5a)
    
    computeTFIDF(sample20, f2, f3b, f5b)
    
    print ("pes mou ta samples--sample80: "+str(len(sample80))+", sample20: "+str(len(sample20)))
    
    for i in idsString:
        f4.write(str(i)+' '+str(idsString.get(i))+'\n')
    
    f.close()
    f1.close()
    f2.close()
    f3a.close()
    f3b.close()
    f3c.close()
    f3d.close()
    f3e.close()
    f4.close()
    
    print ('Outliers from '+str(activeUsersCounter)+' users'+': '+str(standardOutliers))
    print ('Total outliers: '+str(extraOutliers+standardOutliers))
    print ('Total points from '+str(activeUsersCounter)+' users: '+str(totalPointsChecked))
    print ('Average clusters per user: '+str(averageClustersPerUser/activeUsersCounter))
    print ('Average reviews per user: '+str(totalPointsChecked/activeUsersCounter))
    elapsedTime = time.time() - startTimeOverall
    print ('Total time is: '+'%.2f' %(elapsedTime/60)+" minutes.") 
    print ('TA APOTELESMATA EINAI: '+str(counter0)+', '+str(counter50)+', '+str(counter100)+', '+str(counter200)+', '+str(counter300)+', '+str(counter400))

                    