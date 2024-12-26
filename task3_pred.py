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

feature_cols = ['report_date_ts', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos', 'is_weekend', 'is_holiday'] + \
                [f'total_purchase_amt_{i}' for i in range(1, 15)] + \
                [f'tBalance_{i}' for i in range(1, 15)] + \
                [f'yBalance_{i}' for i in range(1, 15)] + \
                [f'direct_purchase_amt_{i}' for i in range(1, 15)] + \
                [f'purchase_bal_amt_{i}' for i in range(1, 15)] + \
                [f'purchase_bank_amt_{i}' for i in range(1, 15)] + \
                [f'total_redeem_amt_{i}' for i in range(1, 15)] + \
                [f'consume_amt_{i}' for i in range(1, 15)] + \
                [f'transfer_amt_{i}' for i in range(1, 15)] + \
                [f'tftobal_amt_{i}' for i in range(1, 15)] + \
                [f'tftocard_amt_{i}' for i in range(1, 15)] + \
                [f'share_amt_{i}' for i in range(1, 15)] + \
                [f'category1_{i}' for i in range(1, 15)] + \
                [f'category2_{i}' for i in range(1, 15)] + \
                [f'category3_{i}' for i in range(1, 15)] + \
                [f'category4_{i}' for i in range(1, 15)] + \
                [f'mfd_daily_yield_{i}' for i in range(1, 15)] + \
                [f'mfd_7daily_yield_{i}' for i in range(1, 15)] + \
                [f'Interest_O_N_{i}' for i in range(1, 15)] + \
                [f'Interest_1_W_{i}' for i in range(1, 15)] + \
                [f'Interest_2_W_{i}' for i in range(1, 15)] + \
                [f'Interest_1_M_{i}' for i in range(1, 15)] + \
                [f'Interest_3_M_{i}' for i in range(1, 15)] + \
                [f'Interest_6_M_{i}' for i in range(1, 15)] + \
                [f'Interest_9_M_{i}' for i in range(1, 15)] + \
                [f'Interest_1_Y_{i}' for i in range(1, 15)]

target_cols = [
    'total_purchase_amt', 'tBalance', 'yBalance', 'direct_purchase_amt', 
    'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt', 'consume_amt', 
    'transfer_amt', 'tftobal_amt', 'tftocard_amt', 'share_amt', 'category1',
    'category2', 'category3', 'category4', 'mfd_daily_yield', 
    'mfd_7daily_yield', 'Interest_O_N', 'Interest_1_W',
    'Interest_2_W', 'Interest_1_M', 'Interest_3_M', 'Interest_6_M', 
    'Interest_9_M', 'Interest_1_Y'
]

already_trained_model = {}

models_dir = "d:/金融大数据处理技术作业/lab4/models20241222194927"

for target_col in target_cols:
    model_path = os.path.join(models_dir, f"linear_regression_model_{target_col}")
    try:
        lr_model = LinearRegressionModel.load(model_path)
        already_trained_model[target_col] = lr_model
        print(f"Loaded model for {target_col} from {model_path}")
    except Exception as e:
        print(f"Error loading model for {target_col} from {model_path}: {e}")


df_existing = spark.read.csv(f"D:/金融大数据处理技术作业/lab4/output/filled_predictions/20241225182101/part-00000-adcff49f-0656-4661-aece-7aeccc0dd74d-c000.csv", header=True, inferSchema=True)


start_date = datetime.strptime("2014-09-01", "%Y-%m-%d")
end_date = datetime.strptime("2014-09-30", "%Y-%m-%d")
current_date = start_date

while current_date <= end_date:

    report_date_str = current_date.strftime("%Y-%m-%d")
    df_last = df_existing.filter(F.col("report_date") == current_date.strftime("%Y-%m-%d"))


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
            F.when(F.col("report_date") == report_date_str, F.lit(prediction_value))
            .otherwise(F.col(target_col))
        )
        for i in range(1, 15):
            future_date = current_date + timedelta(days=i)
            future_date_str = future_date.strftime("%Y-%m-%d")
            future_col = f"{target_col}_{i}"
            if future_col not in df_existing.columns:
                raise ValueError(f"Expected column '{future_col}' not found in df_existing.")
            df_existing = df_existing.withColumn(
                future_col,
                F.when(F.col("report_date") == future_date_str, F.lit(prediction_value))
                .otherwise(F.col(future_col))
            )

    current_date += timedelta(days=1)

output_path = f"d:/金融大数据处理技术作业/lab4/output/filled_predictions/{current_time}"
df_existing.write.csv(output_path, header=True, mode='overwrite')

spark.stop()