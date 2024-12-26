from pyspark.sql import SparkSession
from pyspark.sql.functions import col, format_number, date_format


spark = SparkSession.builder \
    .appName("Task3extract") \
    .config("spark.driver.memory", "30g") \
    .config("spark.executor.memory", "30g") \
    .config("spark.executor.instances", "1") \
    .getOrCreate()

df = spark.read.csv('D:/金融大数据处理技术作业/lab4/output/filled_predictions/20241226193702/part-00000-fc8077b8-7951-4110-88b3-0dafc2b5ea9b-c000.csv', header=True, inferSchema=True)

filtered_df = df.filter((col('report_date') >= 20140901) & (col('report_date') <= 20140930))

selected_columns = filtered_df.select('report_date', 'total_purchase_amt', 'total_redeem_amt')


selected_columns = selected_columns.withColumn(
    "total_purchase_amt", format_number("total_purchase_amt", 2)
)
selected_columns = selected_columns.withColumn(
    "total_redeem_amt", format_number("total_redeem_amt", 2)
)

selected_columns = filtered_df.select(
    col('report_date').cast('bigint').alias('report_date'),
    col('total_purchase_amt').cast('bigint').alias('purchase'),
    col('total_redeem_amt').cast('bigint').alias('redeem')
)

selected_columns.write.csv('D:/金融大数据处理技术作业/lab4/output/final_results_try_to_improve', header=False, sep=',')
