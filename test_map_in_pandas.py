import os, sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.appName("Test").config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1").getOrCreate()

df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])

def my_func(iterator):
    for pdf in iterator:
        yield pdf

res = df.mapInPandas(my_func, schema=df.schema)
res.show()
spark.stop()
