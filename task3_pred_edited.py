from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import time
from pyspark.sql.window import Window
import os
from pyspark.ml.regression import LinearRegressionModel
from pyspark.sql import Row
from pyspark.sql.types import StructField, StructType, StringType, LongType

current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

spark = SparkSession.builder \
    .appName("Task3Test") \
    .config("spark.driver.memory", "30g") \
    .config("spark.executor.memory", "30g") \
    .config("spark.executor.instances", "1") \
    .getOrCreate()

feature_cols = ['report_date', 'weekday_sin', 'month_sin', 'is_weekend', 'is_holiday'] + \
                [f'total_purchase_amt_{i}' for i in range(1, 15)] + \
                [f'total_redeem_amt_{i}' for i in range(1, 15)] + \
                [f'mfd_daily_yield_{i}' for i in range(1, 15)] + \
                [f'Interest_O_N_{i}' for i in range(1, 15)]

target_cols= ['total_purchase_amt','total_redeem_amt','mfd_daily_yield', 'Interest_O_N']
already_trained_model = {}

models_dir = "D:/金融大数据处理技术作业/lab4/models20241226181705"

for target_col in target_cols:
    model_path = os.path.join(models_dir, f"linear_regression_model_{target_col}")
    try:
        lr_model = LinearRegressionModel.load(model_path)
        already_trained_model[target_col] = lr_model
        print(f"Loaded model for {target_col} from {model_path}")
    except Exception as e:
        print(f"Error loading model for {target_col} from {model_path}: {e}")


df_existing = spark.read.csv(f"D:/金融大数据处理技术作业/lab4/output/filled_predictions/20241226191316/part-00000-9d343c0c-cff5-440f-ac1b-779d384a1818-c000.csv", header=True, inferSchema=True)

df_existing = df_existing.withColumn('report_date', F.col('report_date').cast('int'))

start_date = 20140921
end_date = 20140930
current_date = start_date

while current_date <= end_date:

    df_last = df_existing.filter(F.col("report_date") == current_date)


    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    feature_df = assembler.transform(df_last)

    current_predictions = {}

    for target_col in target_cols:
        
        model = already_trained_model.get(target_col)
        prediction = model.transform(feature_df)
        current_predictions[target_col] = prediction
        prediction_value = prediction.select("prediction").collect()[0][0]
        if target_col not in df_existing.columns:
            raise ValueError(f"Expected column '{target_col}' not found in df_existing.")
        df_existing = df_existing.withColumn(
            target_col,
            F.when(F.col("report_date") == current_date, F.lit(prediction_value))
            .otherwise(F.col(target_col))
        )
        for i in range(1, 15):
            future_date = current_date + i
            future_col = f"{target_col}_{i}"
            if future_col not in df_existing.columns:
                raise ValueError(f"Expected column '{future_col}' not found in df_existing.")
            df_existing = df_existing.withColumn(
                future_col,
                F.when(F.col("report_date") == future_date, F.lit(prediction_value))
                .otherwise(F.col(future_col))
            )

    current_date += 1

output_path = f"d:/金融大数据处理技术作业/lab4/output/filled_predictions/{current_time}"
df_existing.write.csv(output_path, header=True, mode='overwrite')

spark.stop()