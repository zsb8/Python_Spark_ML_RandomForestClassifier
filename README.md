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
The best AUC is: 0.7632570521096346, it is better the model which was used pyspark.mllib.tree DecisionTree.
![image](https://user-images.githubusercontent.com/75282285/195732833-3313f096-033d-481b-8d3b-bb638facbb2a.png)


# Spark monitor

![image](https://user-images.githubusercontent.com/75282285/195733514-e300fbb5-b9d8-4322-8aac-a3e83395fa8c.png)

![image](https://user-images.githubusercontent.com/75282285/195733275-dccde786-d75b-4cac-b654-38e4c99313a7.png)



