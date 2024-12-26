from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SimpleApp") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.sql.broadcastTimeout", "1200") \
    .config("spark.network.timeout", "800s") \
    .getOrCreate()

df = spark.read.option("header", "true").csv("D:/LLM_assisted_chip_design/verilogcoder_with_vcd2csv/vcd2csv_dump/Prob079_fsm3onehot.csv")
df.show()

spark.stop()