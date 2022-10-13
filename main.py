from time import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier


def read_data():
    global row_df
    sql_context = SparkSession.builder.getOrCreate()
    path = "hdfs://node1:8020/input/"
    row_df = sql_context.read.format("csv") \
        .option("header", "true") \
        .option("delimiter", "\t") \
        .load(path + "train.tsv")
    print(row_df.count())


def prepare_data():
    def replace_question(x):
        return 0 if x == "?" else float(x)
    replace_question = udf(replace_question)
    df = row_df.select(
               ['url', 'alchemy_category'] +
               [replace_question(col(column)).cast("double").alias(column) for column in row_df.columns[4:] ] )
    train_df, test_df = df.randomSplit([0.7, 0.3])
    return train_df, test_df


def pipeline():
    print("Begin to use StringIndexer.")
    string_indexer = StringIndexer(inputCol="alchemy_category", outputCol="alchemy_category_Index")
    print("Begin to use OneHotEncoder.")
    encoder = OneHotEncoder(dropLast=False,
                            inputCol='alchemy_category_Index',
                            outputCol="alchemy_category_IndexVec")
    print("Begin to use VectorAssember.")
    assembler_inputs = ['alchemy_category_IndexVec'] + row_df.columns[4: -1]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    print("Begin to use RandomForestClassifier.")
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
    result_pipeline = Pipeline(stages=[string_indexer, encoder, assembler, rftvs])
    return result_pipeline


def evaluate_model(model, validation_data):
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                            labelCol="label",
                                            metricName="areaUnderROC")
    predictions = model.transform(validation_data)
    auc = evaluator.evaluate(predictions)
    return auc


if __name__ == "__main__":
    s_time = time()
    print("Reading data stage".center(60, "-"))
    read_data()
    prepare_data()
    train_d, test_d = prepare_data()
    print(test_d.select('url', 'alchemy_category', 'alchemy_category_score', 'is_news', 'label').show(10))
    n_pipeline = pipeline()
    print("Train data stage.")
    pipeline_model = n_pipeline.fit(train_d)
    print("Prediction stage.")
    predicted = pipeline_model.transform(test_d)
    print(predicted.columns)
    print(predicted.select('url', 'features', 'rawprediction', 'probability', 'label', 'prediction').show(10))
    print("Begin to evaluate stage.")
    best_auc = evaluate_model(pipeline_model, test_d)
    print(f"best_auc={best_auc}")







