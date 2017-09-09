import numpy
import os
import math

from sympy.core.numbers import Pi

inputDirClasslabel = "/media/sagar/DATA/Orbica/Work/Dataset/Analysis/ClassLabels/"

outputFile= "/media/sagar/DATA/Orbica/Work/Dataset/Analysis/ML_Data/Featurefile_10Labels.csv"


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

output=open(outputFile,'w')

folder = ["River_0",
"Canal_1",
"Lake_2",
"Pond_3",
"ICE_4",
"IsLand_5",
"Lagoon_6",
"Swamp_7",
"Rapid_8",
"Reservoir_9"]

for subDir in folder:
 subDirPath=os.path.join(inputDirClasslabel,subDir)
 for filename in os.listdir(subDirPath):
    classLabel= subDir.split("_")[1]
    start=True
    tmp=0
    count=0
    gloablCount=0
    uniqueCount=0
    area = 0
    perimeter = 0
    lat = 0
    long = 0
    areaLenRation = 0
    with open(subDirPath+"/"+filename, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            id = (int)(line.split(",")[0])
            if(start):
                tmp=id
                start=False

            line=line.replace("\n","")
            gloablCount = gloablCount + 1

            if (tmp == id):
                area = line.split(",")[1]
                perimeter = (line.split(",")[2])
                lat = line.split(",")[3]
                long = line.split(",")[4]
                areaLenRation = line.split(",")[5]
                count = count + 1

            else:
                #Reference : https://gis.stackexchange.com/questions/20279/how-can-i-calculate-the-average-width-of-a-polygon/181801#181801
                #(Diameter of a circle with the same perimeter as the polygon) * Area / (Area of a circle with the same perimeter as the polygon)
                #diameterwithSameCircle= math.sqrt(area/Pi)*2

                perimeterFloat= (float)(perimeter)
                areaFloat=(float)(area)

                diameterwithSamePerimeter= (perimeterFloat / (2*math.pi))*2
                radius=diameterwithSamePerimeter/2
                areaOfCircleSamePerimeter=(math.pi*(radius*radius))
                avgWidth= diameterwithSamePerimeter *(areaFloat/areaOfCircleSamePerimeter)

                output.write(str(tmp) + "," + str(area) + "," + str(perimeter) + "," + str(areaLenRation) + "," + str(count) + "," +str(avgWidth)+"," +str(classLabel)+"\n")
                tmp = id
                count = 1
                uniqueCount=uniqueCount+1

        perimeterFloat = (float)(perimeter)
        areaFloat = (float)(area)
        diameterwithSamePerimeter = (perimeterFloat / (2 * math.pi)) * 2
        radius = diameterwithSamePerimeter / 2
        areaOfCircleSamePerimeter = (math.pi * (radius * radius))
        avgWidth = diameterwithSamePerimeter * (areaFloat / areaOfCircleSamePerimeter)
        output.write(str(tmp) + "," + str(area) + "," + str(perimeter) + "," + str(areaLenRation) + "," + str(
            count) + "," + str(avgWidth) + "," + str(classLabel) + "\n")
        uniqueCount=uniqueCount+1
        #print("Global Count For" +subDir +" "+str(gloablCount))
        print("Unique Count For "+subDir+" "+str(uniqueCount))
        break

