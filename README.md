# waterBodies_Classification 

Classified Water Bodies Polygons into different categories like River,Canal,Pond,Lake etc using Deep Nerual Networks(AI).

Project Contains below python scripts: 

1)Feature File Generation: 

    DynamicFeatureGeneration.py : Automatically read all the data from different folders and generate featureFile.
    FeatureGeneration_Training.py :Generate Feature File from combined training Vector layer. 
    FeatureGeneration_WaterBodies_Testing.py : Generate Feature file from testing layer.
    
2)Model Training using Deep Neural Networks:

    TrainingModel.py : Train model using neural networks and evaluate its performance/accuracy/confusion matrix.
   
3)Model Testing using pretrained Deep Neural Networks:

    TestingScript.py : Test model and evaluate its performance/accuracy/confusion matrix.
    TestingModel_NZTM.py :Test model on NZTM dataset and evaluate its performance/accuracy/confusion matrix.

