from pyspark.sql import SparkSession
import os
import json


# ========== 1. 初始化Spark ==========
def init_spark():
    os.environ['HADOOP_CONF_DIR'] = '/export/server/hadoop3.3.0/etc/hadoop'
    spark = (SparkSession.builder
             .master("local[1]")
             .appName("export_part00000_all_11")
             .config("spark.hadoop.fs.defaultFS", "hdfs://node1:8020")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ========== 2. 读取指定part文件（适配5.1.2_top10读part-00001） ==========
def read_specified_part(spark, hdfs_folder, part_prefix="part-00000"):
    """
    读取指定前缀的part文件
    :param spark: SparkSession实例
    :param hdfs_folder: HDFS文件夹路径
    :param part_prefix: 要读取的part前缀（如part-00000/part-00001）
    :return: 格式化后的字典列表，空则返回[]
    """
    try:
        # 列出文件夹下指定前缀的所有文件
        file_list = spark.sparkContext.wholeTextFiles(f"{hdfs_folder}/{part_prefix}*").keys().collect()

        if not file_list:
            print(f"WARNING {hdfs_folder}下未找到{part_prefix}开头的文件")
            return []

        # 读取第一个匹配的文件
        df = spark.read.csv(file_list[0], header=True, inferSchema=True)

        if df.count() == 0:
            print(f"WARNING {hdfs_folder}的{part_prefix}文件内容为空")
            return []

        # 格式化浮点数据
        df_pd = df.toPandas()
        for col in df_pd.columns:
            if df_pd[col].dtype == 'float64':
                df_pd[col] = df_pd[col].round(2)

        print(f"SUCCESS 成功读取{hdfs_folder}的{part_prefix}文件（{len(df_pd)}条数据）")
        return df_pd.to_dict('records')
    except Exception as e:
        print(f"WARNING 读取失败：{hdfs_folder} | 原因：{str(e)[:80]}...")
        return []


# ========== 3. 导出JSON文件 ==========
def export_json(data, filename):
    output_dir = '/export/tmp/movie_json'
    os.makedirs(output_dir, exist_ok=True)
    json_path = f"{output_dir}/{filename}"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    status = "SUCCESS" if data else "WARNING"
    print(f"{status} 导出{filename}完成（数据量：{len(data)}）")


# ========== 4. 主执行逻辑 ==========
if __name__ == "__main__":
    spark = init_spark()
    print("SUCCESS Spark初始化完成")

    # 定义所有任务：key=文件夹路径, value=(输出JSON名, 要读取的part前缀)
    hdfs_prefix = "/user/teamhype/processed_data/analysis_result/"
    all_tasks = {
        # 5.1 用户行为分析（4个文件夹）
        f"{hdfs_prefix}5.1.1_rating_distribution.csv": ("5.1.1_rating_dist.json", "part-00000"),
        f"{hdfs_prefix}5.1.2_top10_contribution.csv": ("5.1.2_top10_contribution.json", "part-00001"),  # 重点：读part-00001
        f"{hdfs_prefix}5.1.2_top50_active_users.csv": ("5.1.2_top50_active_users.json", "part-00000"),
        f"{hdfs_prefix}5.1.3_user_genre_preference.csv": ("5.1.3_user_genre_preference.json", "part-00000"),

        # 5.2 电影特征分析（2个文件夹）
        f"{hdfs_prefix}5.2.1_genre_distribution.csv": ("5.2.1_genre_distribution.json", "part-00000"),
        f"{hdfs_prefix}5.2.2_top20_high_score_movies.csv": ("5.2.2_top20_high_score.json", "part-00000"),

        # 5.3 关联特征分析（5个文件夹）
        f"{hdfs_prefix}5.3.1_consistency_distribution.csv": ("5.3.1_consistency_dist.json", "part-00000"),
        f"{hdfs_prefix}5.3.1_user_consistency.csv": ("5.3.1_user_consistency.json", "part-00000"),
        f"{hdfs_prefix}5.3.2_high_score_tags.csv": ("5.3.2_high_score_tags.json", "part-00000"),
        f"{hdfs_prefix}5.3.2_low_score_tags.csv": ("5.3.2_low_score_tags.json", "part-00000"),
        f"{hdfs_prefix}5.3.3_tag_rating_preference.csv": ("5.3.3_tag_rating_preference.json", "part-00000")
    }

    # 执行所有任务
    for hdfs_path, (json_name, part_prefix) in all_tasks.items():
        data = read_specified_part(spark, hdfs_path, part_prefix)
        export_json(data, json_name)

    # 关闭Spark
    spark.stop()
    print("SUCCESS 所有11个文件夹的数据导出任务执行完成")