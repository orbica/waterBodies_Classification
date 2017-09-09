import math

# Provide input & output Path
InputFile="/media/sagar/DATA/Orbica/Work/Dataset/OL017 ECan Waterbodies Classification/Merge_vectorLayers_Training/Nodes_Merged_vectorLayers_Training.csv"
OutputFile="/media/sagar/DATA/Orbica/Work/Dataset/OL017 ECan Waterbodies Classification/Merge_vectorLayers_Training/Model/NodesMerged_FeatureFile.csv"

#write data into file
writer=open(OutputFile,'w')

def featureGenerationWaterBodies(Path):
    start=True
    tmp=0
    startFlag=True
    count=0
    gloablCount=0
    uniqueCount=0
    area = 0
    perimeter = 0
    min=0
    max=0
    ActualLabel=0
    with open(Path, 'r',encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.replace("\n","")
            if(startFlag):
                startFlag=False
                continue
            id = (int)(line.split(",")[6])
            print(id)
            if(start):
                tmp=id
                start=False

            gloablCount = gloablCount + 1
            if (tmp == id):
                area = line.split(",")[1]
                perimeter = line.split(",")[2]
                min = (float)(line.split(",")[3])
                max = (float)(line.split(",")[4])
                ActualLabel=line.split(",")[5]
                count = count + 1
            else:
                if (min < 0):
                    min = 0
                if (max < 0):
                    max = 0

                elev_diff = max - min
                perimeterFloat = (float)(perimeter)
                areaFloat = (float)(area)
                areaLenRation =(float) (areaFloat/perimeterFloat)
                diameterwithSamePerimeter= (perimeterFloat / (2*math.pi))*2
                radius=diameterwithSamePerimeter/2
                areaOfCircleSamePerimeter=(math.pi*(radius*radius))
                avgWidth= diameterwithSamePerimeter *(areaFloat/areaOfCircleSamePerimeter)
                writer.write("%d"% tmp+"," + str(int(min)) + "," + str(int(max)) + "," + str(int(elev_diff)) + "," + str(
                    area) + "," + str(perimeter) +","+ str(areaLenRation) + "," + str(count) + "," +str(avgWidth)+","+ActualLabel+"\n")
                tmp = id
                count = 1
                uniqueCount=uniqueCount+1

        if (min < 0):
            min = 0
        if (max < 0):
            max = 0

        elev_diff = max - min

        perimeterFloat = (float)(perimeter)
        areaFloat = (float)(area)
        areaLenRation = (areaFloat/perimeterFloat)

        diameterwithSamePerimeter = (perimeterFloat / (2 * math.pi)) * 2
        radius = diameterwithSamePerimeter / 2
        areaOfCircleSamePerimeter = (math.pi * (radius * radius))
        avgWidth = diameterwithSamePerimeter * (areaFloat / areaOfCircleSamePerimeter)
        writer.write("%d"% tmp + "," + str(int(min)) + "," + str(int(max)) + "," + str(int(elev_diff)) + "," + str(
            area) + "," + str(perimeter) + ","+str(areaLenRation) + "," + str(count) + "," + str(avgWidth) +","+ActualLabel+"\n")
        print("Global Count For"+str(gloablCount))

# main method
featureGenerationWaterBodies(InputFile)