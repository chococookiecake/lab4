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

current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

spark = SparkSession.builder \
    .appName("Task3Test") \
    .config("spark.driver.memory", "30g") \
    .config("spark.executor.memory", "30g") \
    .config("spark.executor.instances", "1") \
    .getOrCreate()

df_user_balance = spark.read.csv("user_balance_table.csv", header=True, inferSchema=True)
df_user_profile = spark.read.csv("user_profile_table.csv", header=True, inferSchema=True)
df_mfd_yield = spark.read.csv("mfd_day_share_interest.csv", header=True, inferSchema=True)
df_interest_rate = spark.read.csv("mfd_bank_shibor.csv", header=True, inferSchema=True)

df_user_balance_filled = df_user_balance.fillna({
    'tBalance': 0,
    'yBalance': 0,
    'total_purchase_amt': 0,
    'direct_purchase_amt': 0,
    'purchase_bal_amt': 0,
    'purchase_bank_amt': 0,
    'total_redeem_amt': 0,
    'consume_amt': 0,
    'transfer_amt': 0,
    'tftobal_amt': 0,
    'tftocard_amt': 0,
    'share_amt': 0,
    'category1': 0,
    'category2': 0,
    'category3': 0,
    'category4': 0
})

df_user_balance_filled = df_user_balance_filled.withColumn("report_date", F.to_date(F.col("report_date"), "yyyyMMdd"))

df_user_balance = df_user_balance_filled.join(df_user_profile, on='user_id', how='left')

df_user_balance_filled = df_user_balance.fillna({
    'sex':2,
    'city':114514,
    'constellation':"水牛座"
})

df_aggregated = df_user_balance_filled.groupBy("report_date") \
    .agg(
        F.sum("tBalance").alias("tBalance"),
        F.sum("yBalance").alias("yBalance"),
        F.sum("total_purchase_amt").alias("total_purchase_amt"),
        F.sum("direct_purchase_amt").alias("direct_purchase_amt"),
        F.sum("purchase_bal_amt").alias("purchase_bal_amt"),
        F.sum("purchase_bank_amt").alias("purchase_bank_amt"),
        F.sum("total_redeem_amt").alias("total_redeem_amt"),
        F.sum("consume_amt").alias("consume_amt"),
        F.sum("transfer_amt").alias("transfer_amt"),
        F.sum("tftobal_amt").alias("tftobal_amt"),
        F.sum("tftocard_amt").alias("tftocard_amt"),
        F.sum("share_amt").alias("share_amt"),
        F.sum("category1").alias("category1"),
        F.sum("category2").alias("category2"),
        F.sum("category3").alias("category3"),
        F.sum("category4").alias("category4"),
        F.sum(F.when(F.col("sex") == 1, 1).otherwise(0)).alias("male_count"),  # 男性数量
        F.sum(F.when(F.col("sex") == 0, 1).otherwise(0)).alias("female_count")  # 女性数量
    )

df_aggregated = df_aggregated.withColumn(
    "sex_ratio",
    F.when(F.col("female_count") != 0, F.col("male_count") / F.col("female_count")).otherwise(3)
)

df_mfd_yield = df_mfd_yield.withColumnRenamed("mfd_date", "report_date")
df_mfd_yield = df_mfd_yield.withColumn("report_date", F.to_date(F.col("report_date").cast("string"), "yyyyMMdd"))
df_interest_rate = df_interest_rate.withColumnRenamed("mfd_date", "report_date")
df_interest_rate = df_interest_rate.withColumn("report_date", F.to_date(F.col("report_date").cast("string"), "yyyyMMdd"))

df_user_balance = df_aggregated.join(df_mfd_yield, on='report_date', how='left')
df_user_balance = df_user_balance.join(df_interest_rate, on='report_date', how='left')
df_user_balance = df_user_balance.orderBy("report_date")

columns_to_fill = ["mf_daily_yield", "mfd_7daily_yield", "Interest_O_N", "Interest_1_W", "Interest_2_W", "Interest_1_M", "Interest_3_M", "Interest_6_M", "Interest_9_M", "Interest_1_Y"]

window_spec = Window.orderBy("report_date").rowsBetween(Window.unboundedPreceding, 0)

df_user_balance = df_user_balance.select(
    *[
        F.last(column, True).over(window_spec).alias(column) if column in columns_to_fill else column
        for column in df_user_balance.columns
    ]
)

current_columns = df_user_balance.columns

missing_start_date = "2014-09-01"
missing_end_date = "2014-09-30"

date_df = spark.sql(f"""
    SELECT explode(sequence(to_date('{missing_start_date}'), to_date('{missing_end_date}'), interval 1 day)) as report_date
""").orderBy("report_date")

#df_user_balance.write.csv(f"d:/金融大数据处理技术作业/lab4/output/test1{current_time}", header=True)
for col_name in current_columns:
    if col_name != "report_date":
        date_df = date_df.withColumn(col_name, F.lit(0))

#date_df.write.csv(f"d:/金融大数据处理技术作业/lab4/output/test2{current_time}", header=True)

df_user_balance_filled = df_user_balance.union(date_df)
df_user_balance_filled = df_user_balance_filled.coalesce(1)
#df_user_balance_filled.write.csv(f"d:/金融大数据处理技术作业/lab4/output/test3{current_time}", header=True)

df_user_balance_filled = df_user_balance_filled.withColumn("year", F.year("report_date"))
df_user_balance_filled = df_user_balance_filled.withColumn("month", F.month("report_date"))
df_user_balance_filled = df_user_balance_filled.withColumn("day", F.dayofmonth("report_date"))
df_user_balance_filled = df_user_balance_filled.withColumn("weekday", F.dayofweek("report_date"))

df_user_balance_filled = df_user_balance_filled.withColumn("weekday_sin", F.sin(2 * 3.14159 * F.col("weekday") / 7))
df_user_balance_filled = df_user_balance_filled.withColumn("weekday_cos", F.cos(2 * 3.14159 * F.col("weekday") / 7))
df_user_balance_filled = df_user_balance_filled.withColumn("month_sin", F.sin(2 * 3.14159 * F.col("month") / 12))
df_user_balance_filled = df_user_balance_filled.withColumn("month_cos", F.cos(2 * 3.14159 * F.col("month") / 12))

df_user_balance_filled = df_user_balance_filled.withColumn("is_weekend", 
    F.when((F.col("weekday") == 6) | (F.col("weekday") == 7), 1).otherwise(0))

df_user_balance_filled = df_user_balance_filled.withColumn("is_weekend", 
    F.when(F.col("report_date").isin("2014-05-04", "2014-09-28"), 0)
     .otherwise(F.col("is_weekend")))

df_user_balance = df_user_balance_filled.withColumn("is_holiday", 
    F.when(F.col("report_date").isin(
        "2013-09-19", "2013-09-20", "2013-09-21", "2013-10-01", "2013-10-02",
        "2013-10-03", "2013-10-04", "2013-10-05", "2013-10-06", "2013-10-07", "2013-12-24", 
        "2013-12-31", "2014-01-01", "2014-01-23", "2014-01-24", "2014-01-25", "2014-01-26",
        "2014-01-27", "2014-01-28", "2014-01-29", "2014-01-30",
        "2014-01-31", "2014-02-01", "2014-02-02", "2014-02-03",
        "2014-02-04", "2014-02-05", "2014-02-06", "2014-02-07",
        "2014-04-05", "2014-04-06", "2014-04-07",
        "2014-05-01", "2014-05-02", "2014-05-03", "2014-05-31", "2014-06-01", "2014-06-02",
        "2014-09-06", "2014-09-07", "2014-09-08"), 1).otherwise(0))

df_user_balance_with_no_last_date = df_user_balance

for i in range(1, 15):
    df_user_balance = df_user_balance.withColumn(
        f'last_date_{i}', F.date_add(F.col('report_date'), -i)
    )

df_new = df_user_balance

selected_columns = ['report_date','total_purchase_amt', 'tBalance', 'yBalance', 'direct_purchase_amt', 'purchase_bal_amt', 'purchase_bank_amt',
               'total_redeem_amt', 'consume_amt', 'transfer_amt', 'tftobal_amt', 'tftocard_amt', 'share_amt', 'category1',
               'category2', 'category3', 'category4', 'mfd_daily_yield', 'mfd_7daily_yield', 'Interest_O_N', 'Interest_1_W',
               'Interest_2_W', 'Interest_1_M', 'Interest_3_M', 'Interest_6_M', 'Interest_9_M', 'Interest_1_Y']

df_user_balance_with_no_last_date_selected = df_user_balance_with_no_last_date.select(*selected_columns)

for i in range(1, 15):

    columns = df_user_balance_with_no_last_date_selected.columns
    i_df_user_balance_with_no_last_date_selected = df_user_balance_with_no_last_date_selected

    for column in columns:
        if column != "report_date":
             i_df_user_balance_with_no_last_date_selected = i_df_user_balance_with_no_last_date_selected.withColumnRenamed(column, column + f'_{i}')
    
    i_df_user_balance_with_no_last_date_selected = i_df_user_balance_with_no_last_date_selected.withColumnRenamed('report_date', f'last_date_{i}')

    df_new = df_new.join(
        i_df_user_balance_with_no_last_date_selected,
        on=f'last_date_{i}', 
        how='left'
    )

df_save = df_new.withColumn("report_date_ts", F.unix_timestamp("report_date"))
df_save.write.csv(f"d:/金融大数据处理技术作业/lab4/output/test{current_time}", header=True)

df_filtered = df_new.filter(
    (F.col("report_date") >= "2013-07-15") & (F.col("report_date") <= "2014-08-31")
)

df_filtered = df_filtered.withColumn("report_date_ts", F.unix_timestamp("report_date"))

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

target_cols = ['total_purchase_amt', 'tBalance', 'yBalance', 'direct_purchase_amt',
'purchase_bal_amt', 'purchase_bank_amt', 'total_redeem_amt', 'consume_amt', 'transfer_amt', 
'tftobal_amt', 'tftocard_amt', 'share_amt', 'category1', 'category2',
'category3', 'category4', 'mfd_daily_yield',
'mfd_7daily_yield', 'Interest_O_N', 'Interest_1_W',
'Interest_2_W', 'Interest_1_M', 'Interest_3_M', 'Interest_6_M', 
'Interest_9_M', 'Interest_1_Y']

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_features = assembler.transform(df_filtered)

train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=1234)

trained_models = {}

#models_dir = f"d:/金融大数据处理技术作业/lab4/models{current_time}"
models_dir = f"d:/金融大数据处理技术作业/lab4/models20241222194927"


for target_col in target_cols:
    lr = LinearRegression(featuresCol='features', labelCol=target_col, maxIter=50)
    lr_model = lr.fit(train_data)
    print(f"Coefficients for {target_col}: {lr_model.coefficients}")
    print(f"Intercept for {target_col}: {lr_model.intercept}")
    trained_models[target_col] = lr_model
    model_path = os.path.join(models_dir, f"linear_regression_model_{target_col}")
    lr_model.save(model_path)
    print(f"Model for {target_col} saved at {model_path}")
    predictions = lr_model.transform(test_data)
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target_col, metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE) for {target_col}: {rmse}")
    r2_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target_col, metricName="r2")
    r2 = r2_evaluator.evaluate(predictions)
    print(f"R2 for {target_col}: {r2}")
    predictions.select("report_date", target_col, "prediction").show(5)

#df_new.write.csv(f"d:/金融大数据处理技术作业/lab4/output/test{current_time}", header=True)

# df_filtered = df_new.filter(
#     (F.col("report_date") >= "2013-08-05") & (F.col("report_date") <= "2014-08-31")
# )
