from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, sum, when


spark = SparkSession.builder \
    .appName("UserBalanceAnalysis") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.sql.broadcastTimeout", "1200") \
    .config("spark.network.timeout", "800s") \
    .getOrCreate()


df = spark.read.option("header", "true").csv("user_balance_table.csv")

df = df.withColumn("total_purchase_amt", 
                   when(col("total_purchase_amt").rlike(r"^\d+$"), col("total_purchase_amt").cast("int"))
                   .otherwise(0)) \
       .withColumn("total_redeem_amt", 
                   when(col("total_redeem_amt").rlike(r"^\d+$"), col("total_redeem_amt").cast("int"))
                   .otherwise(0)) \
       .withColumn("user_id", col("user_id").cast("int")) \
       .withColumn("report_date", col("report_date").cast("string"))

df = df.na.fill({"total_purchase_amt": 0, "total_redeem_amt": 0})

daily_funds = df.groupBy("report_date").agg(
    sum("total_purchase_amt").alias("total_purchase_amt"),
    sum("total_redeem_amt").alias("total_redeem_amt")
)

daily_funds = daily_funds.orderBy("report_date")

daily_funds.show(truncate=False)

df_with_month = df.withColumn("month", col("report_date").substr(1, 6))

df_august = df_with_month.filter(col("month") == "201408")

user_active_days = df_august.groupBy("user_id").agg(
    countDistinct("report_date").alias("active_days")
)

active_users = user_active_days.filter(col("active_days") >= 5)

active_user_count = active_users.count()

print(f"2014 年 8 月活跃用户总数：{active_user_count}")

spark.stop()