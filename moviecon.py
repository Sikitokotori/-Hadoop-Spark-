import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, trim
from pyspark.sql.types import StringType, FloatType, IntegerType

if __name__ == "__main__":
    # 基础环境配置
    os.environ['JAVA_HOME'] = '/export/server/jdk1.8.0_65'
    os.environ['HADOOP_HOME'] = '/export/server/hadoop3.3.0'
    os.environ['PYSPARK_PYTHON'] = '/root/pyspark_code/bin/python'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/pyspark_code/bin/python'
    os.environ['HADOOP_CONF_DIR'] = '/export/server/hadoop3.3.0/etc/hadoop'

    spark = (SparkSession.builder
        .appName("MovieDataPreprocessing")
        .master("local[2]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.hadoop.fs.defaultFS", "hdfs://node1:8020")
        .config("spark.submit.deployMode", "client")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.port", "4040")
        .getOrCreate())

    # 只输出ERROR日志，屏蔽WARN
    spark.sparkContext.setLogLevel("ERROR")

    # ===================== 定义完整HDFS路径 =====================
    HDFS_BASE_PATH = "hdfs://node1:8020/user/teamhype"
    RAW_RATINGS_PATH = f"{HDFS_BASE_PATH}/raw_data/ratings.csv"
    RAW_TAGS_PATH = f"{HDFS_BASE_PATH}/raw_data/tags.csv"
    RAW_MOVIES_PATH = f"{HDFS_BASE_PATH}/raw_data/movies.csv"
    PREP_RATINGS_PATH = f"{HDFS_BASE_PATH}/processed_data/ratings_parquet"
    PREP_TAGS_PATH = f"{HDFS_BASE_PATH}/processed_data/tags_parquet"
    PREP_MOVIES_PATH = f"{HDFS_BASE_PATH}/processed_data/movies_parquet"

    # ===================== 分步执行 + 实时打印进度 =====================
    print("===== 1. 开始加载评分表数据 =====")
    ratings_df = spark.read.csv(RAW_RATINGS_PATH, header=False, schema="UserID STRING, MovieID STRING, Rating STRING")
    print(f" 评分表原始行数：{ratings_df.count()}")

    print("\n===== 2. 清洗评分表数据（缺失值+异常值） =====")
    ratings_clean_df = (ratings_df
                        # 缺失值处理：删除UserID/MovieID/Rating为空/纯空格的行
                        .filter(col("UserID").isNotNull() & (trim(col("UserID")) != "") &
                                col("MovieID").isNotNull() & (trim(col("MovieID")) != "") &
                                col("Rating").isNotNull() & (trim(col("Rating")) != ""))
                        # 异常值过滤1：Rating必须是数字/小数点组成
                        .filter(col("Rating").rlike("^[0-9.]+$"))
                        # 格式转换：将Rating转为浮点型（便于后续数值计算）
                        .withColumn("Rating", col("Rating").cast(FloatType()))
                        # 异常值过滤2：Rating必须在0-5范围内
                        .filter((col("Rating") >= 0) & (col("Rating") <= 5))
                        # 格式优化：UserID/MovieID转为整型
                        .withColumn("UserID", col("UserID").cast(IntegerType()))
                        .withColumn("MovieID", col("MovieID").cast(IntegerType())))
    print(f" 评分表清洗后行数：{ratings_clean_df.count()}")

    print("\n===== 3. 开始加载标签表数据 =====")
    tags_df = spark.read.csv(RAW_TAGS_PATH, header=False, schema="UserID STRING, MovieID STRING, Tag STRING")
    print(f" 标签表原始行数：{tags_df.count()}")

    print("\n===== 4. 清洗标签表数据 =====")
    tags_clean_df = (tags_df
        .filter(col("UserID").isNotNull() & (trim(col("UserID")) != "") &
                col("MovieID").isNotNull() & (trim(col("MovieID")) != ""))
        .withColumn("Tag", when(col("Tag").isNull() | (trim(col("Tag")) == ""), "unknown").otherwise(trim(col("Tag"))))
        .withColumn("Tag", regexp_replace(col("Tag"), "[^a-zA-Z0-9\\s]", ""))
        .withColumn("UserID", col("UserID").cast(IntegerType()))
        .withColumn("MovieID", col("MovieID").cast(IntegerType())))
    print(f" 标签表清洗后行数：{tags_clean_df.count()}")

    print("\n===== 5. 开始加载电影表数据 =====")
    movies_df = spark.read.csv(RAW_MOVIES_PATH, header=False, schema="MovieID STRING, Title STRING, Genres STRING")
    print(f" 电影表原始行数：{movies_df.count()}")

    print("\n===== 6. 清洗电影表数据 =====")
    movies_clean_df = (movies_df
        .filter(col("MovieID").isNotNull() & (trim(col("MovieID")) != "") &
                col("Title").isNotNull() & (trim(col("Title")) != ""))
        .withColumn("Genres", when(col("Genres").isNull() | (trim(col("Genres")) == ""), "Unknown").otherwise(col("Genres")))
        .withColumn("Genres", regexp_replace(col("Genres"), "[^a-zA-Z|\\s]", ""))
        .withColumn("MovieID", col("MovieID").cast(IntegerType())))
    print(f" 电影表清洗后行数：{movies_clean_df.count()}")

    print("\n===== 7. 输出Parquet格式文件 =====")
    ratings_clean_df.write.mode("overwrite").parquet(PREP_RATINGS_PATH)
    tags_clean_df.write.mode("overwrite").parquet(PREP_TAGS_PATH)
    movies_clean_df.write.mode("overwrite").parquet(PREP_MOVIES_PATH)
    print(f" Parquet文件已输出到：{HDFS_BASE_PATH}/processed_data/")

    # 预览结果
    print("\n===== 最终清洗后评分表示例 =====")
    ratings_clean_df.show(3, truncate=False)

    # 关闭Spark
    spark.stop()
    print("\n所有预处理步骤执行完成！")