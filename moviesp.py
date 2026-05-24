import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
   col, count, avg, desc, sum, abs,
   when, split, explode, countDistinct, round as spark_round
)

if __name__ == "__main__":
   # 1. 环境配置
   os.environ['JAVA_HOME'] = '/export/server/jdk1.8.0_65'
   os.environ['HADOOP_HOME'] = '/export/server/hadoop3.3.0'
   os.environ['PYSPARK_PYTHON'] = '/root/pyspark_code/bin/python'
   os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/pyspark_code/bin/python'
   os.environ['HADOOP_CONF_DIR'] = '/export/server/hadoop3.3.0/etc/hadoop'

   # 2. 初始化SparkSession
   spark = (SparkSession.builder
            .appName("MovieAnalysis")
            .master("local[2]")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.hadoop.fs.defaultFS", "hdfs://node1:8020")
            .getOrCreate())
   spark.sparkContext.setLogLevel("ERROR")

   # 3. 加载数据集
   HDFS_BASE = "hdfs://node1:8020/user/teamhype/processed_data"
   ratings_df = spark.read.parquet(f"{HDFS_BASE}/ratings_parquet")
   movies_df = spark.read.parquet(f"{HDFS_BASE}/movies_parquet")
   tags_df = spark.read.parquet(f"{HDFS_BASE}/tags_parquet")

   # ===================== 通用数据预处理 =====================
   movies_with_genre = (movies_df
                        .withColumn("Genre", explode(split(col("Genres"), "\\|")))
                        .filter(col("Genre") != "")
                        .select("MovieID", "Genre"))

   top2_genres = (movies_with_genre
                  .groupBy("Genre")
                  .agg(count("*").alias("频次"))
                  .orderBy(desc("频次"))
                  .limit(2)
                  .collect())
   genre1 = top2_genres[0]["Genre"] if len(top2_genres) >= 1 else ""
   genre2 = top2_genres[1]["Genre"] if len(top2_genres) >= 2 else ""

   # ===================== 5.1 用户行为分析 =====================
   print("===== 5.1 用户行为分析 =====")

   # 5.1.1 用户整体评分分布
   print("----- 5.1.1 用户整体评分分布 -----")
   rating_distribution = (ratings_df
                          .groupBy("Rating")
                          .agg(
       count("*").alias("评分频次"),
       spark_round(count("*") / ratings_df.count() * 100, 2).alias("占比(%)")
   )
                          .orderBy("Rating"))
   rating_distribution.show()
   rating_distribution.write.mode("overwrite").csv(
       f"{HDFS_BASE}/analysis_result/5.1.1_rating_distribution.csv", header=True)

   # 5.1.2 高活跃用户识别
   print("----- 5.1.2 高活跃用户识别 -----")
   user_rating_count = (ratings_df
                        .groupBy("UserID")
                        .agg(count("*").alias("评分次数"))
                        .orderBy(desc("评分次数")))
   top50_active_users = user_rating_count.limit(50)
   top50_active_users.show(50)

   total_ratings = ratings_df.count()
   total_users = user_rating_count.count()
   top10_percent_num = max(1, int(total_users * 0.1))
   top10_percent_ratings = user_rating_count.limit(top10_percent_num).agg(sum("评分次数")).first()[0]
   top10_contribution = round(top10_percent_ratings / total_ratings * 100, 2)
   print(f"前10%用户（{top10_percent_num}人）贡献了 {top10_contribution}% 的评分")

   top50_active_users.write.mode("overwrite").csv(
       f"{HDFS_BASE}/analysis_result/5.1.2_top50_active_users.csv", header=True)
   spark.createDataFrame([(top10_percent_num, top10_contribution)],
                         ["前10%用户数量", "前10%用户评分贡献占比(%)"]).write.mode("overwrite").csv(
       f"{HDFS_BASE}/analysis_result/5.1.2_top10_contribution.csv", header=True)

   # 5.1.3 用户的电影类型偏好
   print("----- 5.1.3 用户电影类型偏好 -----")
   top10_user_rows = user_rating_count.limit(10).collect()
   top10_user_ids = [row[0] for row in top10_user_rows]
   print(f"Top10活跃用户ID：{top10_user_ids}")

   user_id_sample = ratings_df.select("UserID").limit(1).collect()[0][0]
   if isinstance(user_id_sample, int):
       top10_user_ids = [int(id) for id in top10_user_ids]
   else:
       top10_user_ids = [str(id) for id in top10_user_ids]

   user_genre_preference = (ratings_df
                            .join(movies_with_genre, on="MovieID", how="inner")
                            .groupBy("UserID", "Genre")
                            .agg(count("*").alias("评分次数"))
                            .orderBy("UserID", desc("评分次数")))

   top10_user_preference = (user_genre_preference
                            .filter(col("UserID").isin(top10_user_ids))
                            .orderBy(col("UserID"), desc("评分次数")))

   distinct_users = top10_user_preference.select("UserID").distinct().count()
   print(f"Top10用户偏好分析中实际包含的用户数：{distinct_users}")
   top10_user_preference.show(100)

   top10_user_preference.write.mode("overwrite").csv(
       f"{HDFS_BASE}/analysis_result/5.1.3_user_genre_preference.csv", header=True)

   # ===================== 5.2 电影特征分析 =====================
   print("\n===== 5.2 电影特征分析 =====")

   # 5.2.1 电影类型分布
   print("----- 5.2.1 电影类型分布 -----")
   genre_distribution = (movies_with_genre
                         .groupBy("Genre")
                         .agg(
       countDistinct("MovieID").alias("电影数量"),
       spark_round(countDistinct("MovieID") / movies_df.count() * 100, 2).alias("占比(%)")
   )
                         .orderBy(desc("电影数量")))
   genre_distribution.show()
   genre_distribution.write.mode("overwrite").csv(
       f"{HDFS_BASE}/analysis_result/5.2.1_genre_distribution.csv", header=True)

   # 5.2.2 高评分电影Top20
   print("----- 5.2.2 高评分电影Top20 -----")
   movie_rating_count = (ratings_df
                         .groupBy("MovieID")
                         .agg(count("*").alias("评分次数")))
   high_score_movies = (ratings_df
                        .groupBy("MovieID")
                        .agg(spark_round(avg("Rating"), 2).alias("平均评分"))
                        .join(movie_rating_count, on="MovieID", how="inner")
                        .filter(col("评分次数") >= 15)
                        .join(movies_df.select("MovieID", "Title"), on="MovieID", how="inner")
                        .orderBy(desc("平均评分"))
                        .limit(20))
   high_score_movies.show(truncate=False)
   high_score_movies.write.mode("overwrite").csv(
       f"{HDFS_BASE}/analysis_result/5.2.2_top20_high_score_movies.csv", header=True)

   # ===================== 5.3 关联特征分析 =====================
   print("\n===== 5.3 关联特征分析 =====")

   # 5.3.1 用户评分偏好一致性
   print(f"----- 5.3.1 用户评分偏好一致性（{genre1} vs {genre2}） -----")
   if genre1 and genre2:
       genre1_movies = movies_with_genre.filter(col("Genre") == genre1).select("MovieID").withColumnRenamed("MovieID",
                                                                                                            f"{genre1}_MovieID")
       genre2_movies = movies_with_genre.filter(col("Genre") == genre2).select("MovieID").withColumnRenamed("MovieID",
                                                                                                            f"{genre2}_MovieID")

       user_genre1_rating = (ratings_df
                             .join(genre1_movies, col("MovieID") == col(f"{genre1}_MovieID"), how="inner")
                             .groupBy("UserID")
                             .agg(spark_round(avg("Rating"), 2).alias(f"{genre1}_平均评分")))
       user_genre2_rating = (ratings_df
                             .join(genre2_movies, col("MovieID") == col(f"{genre2}_MovieID"), how="inner")
                             .groupBy("UserID")
                             .agg(spark_round(avg("Rating"), 2).alias(f"{genre2}_平均评分")))

       user_preference_consistency = (user_genre1_rating
                                      .join(user_genre2_rating, on="UserID", how="inner")
                                      .withColumn("评分差值",
                                                  abs(col(f"{genre1}_平均评分") - col(f"{genre2}_平均评分")))
                                      .withColumn("一致性等级",
                                                  when(col("评分差值") < 0.5, "高度一致")
                                                  .when(col("评分差值") < 1.0, "较一致")
                                                  .otherwise("不一致")))

       consistency_distribution = (user_preference_consistency
                                   .groupBy("一致性等级")
                                   .agg(count("*").alias("用户数量")))
       consistency_distribution.show()

       user_preference_consistency.write.mode("overwrite").csv(
           f"{HDFS_BASE}/analysis_result/5.3.1_user_consistency.csv", header=True)
       consistency_distribution.write.mode("overwrite").csv(
           f"{HDFS_BASE}/analysis_result/5.3.1_consistency_distribution.csv", header=True)
   else:
       print("数据中无足够的电影类型，跳过一致性分析")

   # 5.3.2 高分/低分电影的标签分布差异
   print("----- 5.3.2 高分/低分电影标签分布 -----")
   movie_score = (ratings_df
                  .groupBy("MovieID")
                  .agg(avg("Rating").alias("平均评分")))
   high_score_movie_ids = movie_score.filter(col("平均评分") >= 4.0).select("MovieID")
   low_score_movie_ids = movie_score.filter(col("平均评分") <= 2.0).select("MovieID")

   # 标签表去重（同一电影+同一标签只保留一条）
   tags_df_distinct = tags_df.select("MovieID", "Tag").distinct().filter(col("Tag") != "")

   high_score_tags = (tags_df_distinct
                      .join(high_score_movie_ids, on="MovieID", how="inner")
                      .groupBy("Tag")
                      .agg(count("*").alias("标签频次"))
                      .orderBy(desc("标签频次"))
                      .limit(10))
   low_score_tags = (tags_df_distinct
                     .join(low_score_movie_ids, on="MovieID", how="inner")
                     .groupBy("Tag")
                     .agg(count("*").alias("标签频次"))
                     .orderBy(desc("标签频次"))
                     .limit(10))

   print("高分电影Top10标签：")
   high_score_tags.show(truncate=False)
   print("低分电影Top10标签：")
   low_score_tags.show(truncate=False)

   high_score_tags.write.mode("overwrite").csv(
       f"{HDFS_BASE}/analysis_result/5.3.2_high_score_tags.csv", header=True)
   low_score_tags.write.mode("overwrite").csv(
       f"{HDFS_BASE}/analysis_result/5.3.2_low_score_tags.csv", header=True)

   # 5.3.3 用户对作品标签的评分偏好
   print("----- 5.3.3 用户对标签的评分偏好 -----")
   tags_df_distinct = tags_df.select("MovieID", "Tag").distinct().filter(col("Tag") != "")
   print(f"标签表去重后的数据量：{tags_df_distinct.count()}")

   # 先计算每部电影的平均评分（避免同一电影多次评分导致重复计算）
   movie_avg_rating = (ratings_df
                       .groupBy("MovieID")
                       .agg(spark_round(avg("Rating"), 2).alias("电影平均评分")))
   print(f"电影平均评分数据量：{movie_avg_rating.count()}")

   # 关联电影平均评分和去重后的标签表
   tag_movie_rating = (tags_df_distinct
                       .join(movie_avg_rating, on="MovieID", how="inner")
                       .filter(col("Tag") != ""))
   print(f"标签+电影评分关联后的数据量：{tag_movie_rating.count()}")


   tag_rating_preference = (tag_movie_rating
                            .groupBy("Tag")
                            .agg(
       spark_round(avg("电影平均评分"), 2).alias("标签平均评分"),  # 按标签计算电影平均评分的均值
       countDistinct("MovieID").alias("关联电影数量"),
       count("*").alias("总评分次数")
   )
                            .filter(col("关联电影数量") >= 10)  # 过滤小众标签（关联电影<10）
                            .orderBy(desc("标签平均评分")))

   # 打印前20个标签
   print("标签评分偏好Top20：")
   tag_rating_preference.show(20, truncate=False)

   # 结果保存
   tag_rating_preference.write.mode("overwrite").csv(
       f"{HDFS_BASE}/analysis_result/5.3.3_tag_rating_preference.csv", header=True)

   # 关闭Spark
   spark.stop()
   print(f"\n所有分析任务执行完成！")
   print(f"结果已保存到 HDFS：{HDFS_BASE}/analysis_result/")