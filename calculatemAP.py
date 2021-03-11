from tabulate import tabulate
from helper import CustomTensors
from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import *
import operator

class MeanAveragePrecision:

    def __init__(self,category_index):
        self.allBoundingBoxes = BoundingBoxes()
        self.category_index = category_index
        
    def AddGroundTruth(self,row):
        #for row in groundTruthList:
        bb = BoundingBox(
                    row.filename,
                    row.classname,
                    row.xmin,
                    row.ymin,
                    row.xmax,
                    row.ymax,
                    CoordinatesType.Absolute, 
                    (row.height, row.width),
                    BBType.GroundTruth,
                    format=BBFormat.XYX2Y2)
        self.allBoundingBoxes.addBoundingBox(bb)

    def AddDetectedVales(self,row,confidence): 
        #for row in detectedValuesList:
        bb = BoundingBox(
            row.filename,
            row.classname,
            row.xmin,
            row.ymin,
            row.xmax,
            row.ymax,
            CoordinatesType.Absolute, (row.height, row.width),
            BBType.Detected,
            confidence,
            format=BBFormat.XYX2Y2)
        self.allBoundingBoxes.addBoundingBox(bb)
        
    def __call__(self,gtList,detections):
        detectionBoxes = detections['detection_boxes'][0].numpy()
        detectionClasses =  (detections['detection_classes'][0].numpy() + 1).astype(int)  #Adding one is mandatory
        detectionScores =  detections['detection_scores'][0].numpy()
        detectionScores = list(filter(lambda x: x > 0.1, detectionScores))
        for boxNormalised,classs,score in zip(detectionBoxes,detectionClasses,detectionScores):
            className =  self.category_index[classs]["name"]
            groundTruthClass = list(filter(lambda x: x.classname  == className, gtList))
            
            iouList = []
            box = []
            for j in groundTruthClass:
                ymin = boxNormalised[0]*j.height
                xmin = boxNormalised[1]*j.width
                ymax = boxNormalised[2]*j.height
                xmax = boxNormalised[3]*j.width
                box = [xmin,ymin,xmax,ymax]
                iou = Evaluator.iou(box, [j.xmin,j.ymin,j.xmax,j.ymax])
                iouList.append(iou)
            if(len(iouList) > 0):
                index, value = max(enumerate(iouList), key=operator.itemgetter(1))
                gtTensor = groundTruthClass[index]
            else:
                continue  #it is a case of true negative. hence ignored.
                #gtTensor = groundTruthClass[0]
            self.AddGroundTruth(gtTensor)
            self.AddDetectedVales(CustomTensors(gtTensor.filename,gtTensor.width,gtTensor.height,className,int(box[0]),int(box[1]),int(box[2]),int(box[3])),score)
        print("done")

    def CalculatemAP(self):
        evaluator = Evaluator()
        iouThresholds = [.25,.5,.75,.9]
        mAPTable = []
        for iouThreshold in iouThresholds:
            metricsPerClass = evaluator.GetPascalVOCMetrics(
                self.allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
                IOUThreshold=iouThreshold,  # IOU threshold
                method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code

            print("Average precision values per class:")
            # Loop through classes to obtain their metrics
            averagePrecision = []
            for mc in metricsPerClass:
                # Get metric values per each class
                c = mc['class']
                precision = mc['precision']
                recall = mc['recall']
                average_precision = mc['AP']
                ipre = mc['interpolated precision']
                irec = mc['interpolated recall']
                # Print AP per class
                print('%s: %f' % (c, average_precision))
                averagePrecision.append(average_precision)
            print("Mean Average Precision @"+ str(int(iouThreshold*100)) +" is:" + str(sum(averagePrecision)/len(averagePrecision)))
            mAPTable.append([int(iouThreshold*100),sum(averagePrecision)/len(averagePrecision)])
        print(tabulate(mAPTable,headers=["IOU Threshold","Mean Average Precision"],tablefmt="fancy_grid"))
