import math


InputFile="/media/sagar/DATA/Orbica/Work/Dataset/OL017 ECan Waterbodies Classification/NZTM_Wateroutline_Clipped_SingleFeature/NZTM_WaterOutlines_Clipped_SingleFeature_9Sep.csv"
OutputFile="/media/sagar/DATA/Orbica/Work/Dataset/OL017 ECan Waterbodies Classification/NZTM_Wateroutline_Clipped_SingleFeature/NZTM_WaterOutlines_Clipped_SingleFeature_9SepModel.csv"

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
    ActualLabel=""
    with open(Path, 'r',encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.replace("\n","")
            if(startFlag):
                startFlag=False
                continue
            id = (int)(line.split(",")[25])
            print(id)
            if(start):
                tmp=id
                start=False

            gloablCount = gloablCount + 1
            if (tmp == id):
                perimeter = line.split(",")[11]
                area = line.split(",")[12]
                min = (float)(line.split(",")[23])
                max = (float)(line.split(",")[24])
                ActualLabel=line.split(",")[3]
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
                if(ActualLabel=="RAPID"):
                    ActualLabel="RIVER"
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
        if (ActualLabel == "RAPID"):
            ActualLabel = "RIVER"
        writer.write("%d"% tmp + "," + str(int(min)) + "," + str(int(max)) + "," + str(int(elev_diff)) + "," + str(
            area) + "," + str(perimeter) + ","+str(areaLenRation) + "," + str(count) + "," + str(avgWidth) +","+ActualLabel+"\n")
        print("Global Count For"+str(gloablCount))




def featureGenerationWaterBodies_NZTM(Path):
    start=True
    tmp=0
    startFlag=True
    count=0
    gloablCount=0
    uniqueCount=0
    area = 0
    perimeter = 0
    with open(Path, 'r',encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.replace("\n","")
            if(startFlag):
                startFlag=False
                continue
            id = (int)(line.split(",")[16])
            print(id)

            if(start):
                tmp=id
                start=False

            gloablCount = gloablCount + 1

            if (tmp == id):

                perimeter = line.split(",")[8]
                area = line.split(",")[9]
                min = (float)(line.split(",")[13])
                max = (float)(line.split(",")[14])
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
                    area) + "," + str(perimeter) +","+ str(areaLenRation) + "," + str(count) + "," +str(avgWidth)+"\n")
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
            area) + "," + str(perimeter) + ","+str(areaLenRation) + "," + str(count) + "," + str(avgWidth) + "\n")
        print("Global Count For"+str(gloablCount))


#Feature File with Ecan WaterBodies

featureGenerationWaterBodies(InputFile)

# Feature File NZTM

featureGenerationWaterBodies_NZTM(InputFile)
