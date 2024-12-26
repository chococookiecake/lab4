from pyspark.sql import SparkSession
from pyspark.sql.functions import col, format_number, date_format


spark = SparkSession.builder \
    .appName("Task3extract") \
    .config("spark.driver.memory", "30g") \
    .config("spark.executor.memory", "30g") \
    .config("spark.executor.instances", "1") \
    .getOrCreate()

df = spark.read.csv('D:/金融大数据处理技术作业/lab4/output/filled_predictions/20241226093331/part-00000-0addfcdb-df8e-4bfb-849b-b649ddf6dc60-c000.csv', header=True, inferSchema=True)

df = df.withColumn('report_date', col('report_date').cast('date'))
filtered_df = df.filter((col('report_date') >= '2014-09-01') & (col('report_date') <= '2014-09-30'))


selected_columns = filtered_df.select('report_date', 'total_purchase_amt', 'total_redeem_amt')


selected_columns = selected_columns.withColumn(
    "total_purchase_amt", format_number("total_purchase_amt", 2)
)
selected_columns = selected_columns.withColumn(
    "total_redeem_amt", format_number("total_redeem_amt", 2)
)

selected_columns = filtered_df.select(
    date_format(col('report_date'), 'yyyyMMdd').alias('report_date'),
    col('total_purchase_amt').cast('bigint').alias('purchase'),
    col('total_redeem_amt').cast('bigint').alias('redeem')
)

selected_columns.write.csv('D:/金融大数据处理技术作业/lab4/output/final_results_try_to_improve', header=False, sep=',')
