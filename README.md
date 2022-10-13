Data is pubic.
# Python_Spark_ML_RandomForestClassifier
Use RandomForest to find the best AUC


Running environment is Spark + Hadoop + PySpark    
Used the algorithm is RandomForestClassifier.     
Used the library is pyspark.ml.    

# Stage1:  Read data
Placed the tsv on hadoop. Built 3 data sets: (1) Train data, (3) test data.

# Stage2: Find the best model
~~~
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
    param_grid = ParamGridBuilder() \
        .addGrid(rf.impurity, ["gini", "entropy"]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .addGrid(rf.maxBins, [10, 15, 20]) \
        .addGrid(rf.numTrees, [10, 20, 30]) \
        .build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                              labelCol="label",
                                              metricName="areaUnderROC")
    rftvs = TrainValidationSplit(estimator=rf, evaluator=evaluator,
                                 estimatorParamMaps=param_grid, trainRatio=0.8)
~~~

# Stage3: Evaluate
Calculated the AUC using test data set. 
~~~
def evaluate_model(model, validation_data):
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                            labelCol="label",
                                            metricName="areaUnderROC")
    predictions = model.transform(validation_data)
    auc = evaluator.evaluate(predictions)
    return auc
~~~
![image](https://user-images.githubusercontent.com/75282285/195732833-3313f096-033d-481b-8d3b-bb638facbb2a.png)


# Spark monitor
![image](https://user-images.githubusercontent.com/75282285/192587362-ac4c79f9-f87c-4da9-9acc-b67412eb2fa5.png)
![image](https://user-images.githubusercontent.com/75282285/192587799-e3b653f6-4d73-4b33-8126-a1debb838366.png)
![image](https://user-images.githubusercontent.com/75282285/192587445-b66c945a-929d-4b42-80c5-5ab5df2d35c1.png)

# DebugString
Print the DebugString.
~~~
print(model.toDebugString())
~~~

![image](https://user-images.githubusercontent.com/75282285/192617611-b294921c-5be5-4393-9073-96793e3c46b4.png)
