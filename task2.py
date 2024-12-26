from pyspark.sql import SparkSession
from pyspark.sql.functions import col, format_number

spark = SparkSession.builder \
    .appName("UserBalanceAnalysis") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.sql.broadcastTimeout", "1200") \
    .config("spark.network.timeout", "800s") \
    .getOrCreate()

user_profile_df = spark.read.option("header", "true").csv("user_profile_table.csv")

user_balance_df = spark.read.option("header", "true").csv("user_balance_table.csv")

user_balance_df = user_balance_df.withColumn(
    "total_flow", 
    col("total_purchase_amt").cast("int") + col("total_redeem_amt").cast("int")
)

user_profile_df.createOrReplaceTempView("user_profile_table")
user_balance_df.createOrReplaceTempView("user_balance_table")

avg_balance_sql = """
    SELECT u.city AS city_id, AVG(b.tBalance) AS avg_balance
    FROM user_balance_table b
    JOIN user_profile_table u
    ON b.user_id = u.user_id
    WHERE b.report_date = '20140301'
    GROUP BY u.city
    ORDER BY avg_balance DESC
"""
avg_balance_df = spark.sql(avg_balance_sql)
avg_balance_df.show(truncate=False)

user_flow_sql = """
    SELECT p.city AS city_id, b.user_id, SUM(b.total_purchase_amt + b.total_redeem_amt) AS total_flow
    FROM user_balance_table b
    JOIN user_profile_table p
    ON b.user_id = p.user_id
    WHERE b.report_date LIKE '201408%'
    GROUP BY b.user_id, p.city
"""

user_flow_df = spark.sql(user_flow_sql)
user_flow_df.createOrReplaceTempView("user_flow_table")

top_3_users_sql = """
    SELECT city_id, user_id, total_flow
    FROM (
        SELECT city_id, user_id, total_flow,
               ROW_NUMBER() OVER (PARTITION BY city_id ORDER BY total_flow DESC) AS rank
        FROM user_flow_table
    ) ranked
    WHERE rank <= 3
"""
top_3_users_df = spark.sql(top_3_users_sql)

top_3_users_df = top_3_users_df.withColumn("total_flow", format_number("total_flow", 0))

top_3_users_df.show(top_3_users_df.count(), truncate=False)

spark.stop()
