
**Big Data** is very large and complex data that traditional tools can't handle easily.

**Three key aspects of Big Data:**

1. **Volume:** Huge amounts of data.
2. **Velocity:** Data is generated and processed very fast.
3. **Variety:** Different types of data (text, images, videos, etc.).

---
Here’s a brief comparison of **Data Lake** vs **Data Warehouse**:

| Aspect        | Data Lake                                  | Data Warehouse                            |
| ------------- | ------------------------------------------ | ----------------------------------------- |
| **Data Type** | Stores raw, unstructured & structured data | Stores processed, structured data         |
| **Schema**    | Schema-on-read (flexible)                  | Schema-on-write (predefined schema)       |
| **Purpose**   | For big data analytics, machine learning   | For business intelligence and reporting   |
| **Cost**      | Generally lower storage cost               | Higher cost due to processing & structure |
| **Users**     | Data scientists, engineers                 | Business analysts, decision-makers        |
----

Two simple key differences between Hadoop 2 and Hadoop 3:

1. **Storage Efficiency:**

   * Hadoop 2 stores data by making multiple copies (replication).
   * Hadoop 3 uses a smarter method called **Erasure Coding** that saves storage space.

2. **Namenode Support:**

   * Hadoop 2 allows only one main Namenode (the master).
   * Hadoop 3 can have multiple active Namenodes, which helps handle bigger clusters better.
   
---

### What is Hive Metastore? (3 marks)

The **Hive Metastore** is a central repository that stores metadata about Hive tables, such as:

* Table names and their schema (columns, data types)
* Location of data files in HDFS
* Partition information
* Table properties and statistics

It helps Hive efficiently manage and query large datasets by keeping track of all this metadata separately from the actual data.

---

### Can HBase (NoSQL) be configured as Hive Metastore? (1 mark)

No, **HBase cannot be used as the Hive Metastore**.
Hive Metastore typically uses relational databases like MySQL, PostgreSQL, or Derby to store metadata. HBase is a NoSQL database and is not designed to be used as the Hive Metastore backend.

---

Using an example, depict how MapReduce computes word count. 

MapReduce Word Count in brief:

* **Map:** Reads input text, outputs (word, 1) for each word.
* **Shuffle:** Groups all (word, 1) pairs by word.
* **Reduce:** Sums counts for each word.

Example input: `"hello world hello"`
Output:

```
hello 2  
world 1
```

----
**CAP Theorem:**
It states that a distributed system can only guarantee two out of the three properties simultaneously:

* **Consistency (C):** Every read receives the most recent write or an error.
* **Availability (A):** Every request receives a response (without guarantee that it contains the most recent write).
* **Partition Tolerance (P):** The system continues to operate despite network partitions or communication failures.

**MongoDB in CAP Theorem:**
MongoDB is **AP (Availability and Partition tolerance)** by default but can be tuned to favor **Consistency** depending on configuration (like write concern and read preference). Usually, it prioritizes availability and partition tolerance over strict consistency.



Here’s a detailed explanation with simple logical diagrams for each topic you asked about: Hive architecture, Spark architecture, Kafka, and NoSQL types.

## 1. Hive Architecture

```
User
  |
  v
HiveServer2  <-- Client connects here
  |
  v
Driver (Query Compiler & Optimizer)
  |
  v
Metastore (Metadata - table schema, location)
  |
  v
Execution Engine (MapReduce / Spark)
  |
  v
HDFS (Data stored here)
  |
  v
Results back to User
```

**How it works:**

* User submits SQL-like query to HiveServer2.
* Driver compiles and optimizes the query.
* Metastore provides metadata info.
* Query runs using Hadoop’s processing engine (MapReduce or Spark).
* Results sent back to user.

---

## 2. Spark Architecture

```
User’s Spark App (Driver Program)
  |
  v
Cluster Manager (YARN / Mesos / Standalone)
  |
  v
Worker Nodes (Executors)
  |      |       |
  |      |       |-- Run tasks & cache data in memory
  |      |
  |      |-- Run tasks & cache data
  |
  |-- Run tasks & cache data
```

**How it works:**

* Driver program controls the job and sends tasks.
* Cluster Manager allocates resources (workers/executors).
* Executors run tasks and store data in memory for fast processing.

---

## 3. Kafka Logical Architecture

```
Producers --> Kafka Brokers --> Topics (Partitions)
                                  |
                                  v
                             Consumers
```

**Explanation:**

* Producers send messages to Kafka Brokers.
* Messages are stored in topics, split into partitions for scalability.
* Consumers subscribe to topics and read messages at their own pace.

---

## 4. Types of NoSQL Databases

```
+----------------+        +------------------+        +--------------+       +-------------+
| Document Store |        | Key-Value Store  |        | Column Store |       | Graph DB    |
|  (MongoDB)     |        |  (Redis)         |        | (Cassandra)  |       | (Neo4j)     |
+----------------+        +------------------+        +--------------+       +-------------+
| Store JSON-like|        | Store key-value   |        | Store columns|       | Store graphs|
| flexible data  |        | pairs for fast    |        | for big data |       | nodes/edges |
|               |        | lookup & caching  |        | analytics    |       | relationships|
+----------------+        +------------------+        +--------------+       +-------------+
```

---


Write HDFS shell commands for the following-  
1. How do you create a new directory in HDFS with its parent directories (if they don’t 
exist)? (2 marks) 
2. How do you upload multiple files from the local system to a directory in HDFS?. (2 marks) 
3. How do you display the contents of a file in HDFS line by line (paged view)? (2 marks) 
4. How do you list all files and directories in HDFS with detailed information? (2 marks) 
5. How do you delete a directory and its contents in HDFS?  (2marks) 
 
Note: Consider InputDir, OutputDir, XYZ, SampleDir, and file.txt are under the present working 
directory.

# 1. Create a new directory with parent directories (if not exist)
hdfs dfs -mkdir -p /user/hadoop/InputDir/SampleDir

# 2. Upload multiple files from local system to HDFS directory
hdfs dfs -put InputDir/* /user/hadoop/OutputDir/

# 3. Display contents of a file in HDFS line by line (paged view)
hdfs dfs -cat /user/hadoop/XYZ/file.txt | less

# 4. List all files and directories in HDFS with detailed info (recursive)
hdfs dfs -ls -R /user/hadoop/

# 5. Delete a directory and all its contents in HDFS
hdfs dfs -rm -r /user/hadoop/SampleDir

# i. Print version of installed Hadoop
hadoop version

# ii. Copy file1.txt from InputDir to OutputDir as file2.txt
hdfs dfs -cp /user/hadoop/InputDir/file1.txt /user/hadoop/OutputDir/file2.txt

# iii. Delete an empty directory named XYZ
hdfs dfs -rmdir /user/hadoop/XYZ

# iv. List the contents of folder named SampleDir
hdfs dfs -ls /user/hadoop/SampleDir

# v. Fetch usage instructions/details of mkdir command
hdfs dfs -help mkdir

# 1. Print contents of directory with details
hdfs dfs -ls /path/to/directory

# 2. Upload multiple files from local to HDFS directory
hdfs dfs -put file1.txt file2.txt file3.txt /path/to/hdfs/directory

# 3. Display contents of a file in HDFS line by line (paged)
hdfs dfs -cat /path/to/file.txt | less

# 4. Remove file or directory recursively in HDFS
hdfs dfs -rm -r /path/to/file_or_directory

# 5. Copy 'testfile' from HDFS to local file system
hdfs dfs -copyToLocal /path/in/hdfs/testfile /local/path/
```


---

Considering sc as spark content object, and rdd as RDD object, write Spark commands to, 
1. Create an RDD from the following list: List(1, 2, 3, 4, 5,6). (2 marks) 
2. Display/Print first four elements of the RDD. (2 marks) 
3. Display/Print the first element of the RDD. (2 marks) 
4. Display/Print the number of elements in the RDD. (2 marks) 
5. Display sum of all elements of the RDD. (2 marks) 

// 1. Create an RDD from the list List(1, 2, 3, 4, 5, 6)
val rdd = sc.parallelize(List(1, 2, 3, 4, 5, 6))

// 2. Display/Print the first four elements of the RDD
rdd.take(4).foreach(println)

// 3. Display/Print the first element of the RDD
println(rdd.first())

// 4. Display/Print the number of elements in the RDD
println(rdd.count())

// 5. Display the sum of all elements in the RDD
println(rdd.sum())

// 4. Explanation with examples of map and filter transformations

// map: applies a function to each element and returns a new RDD
// Example: multiply each element by 2
val mappedRDD = rdd.map(x => x * 2)
// mappedRDD now contains: 2,4,6,8,10,12,14,16,18,20

// filter: returns a new RDD with elements that satisfy the given condition
// Example: keep only even numbers
val filteredRDD = rdd.filter(x => x % 2 == 0)
// filteredRDD now contains: 2,4,6,8,10


Write a Spark program pseudo-code to load a text file named as text.txt into 
spark RDD and compute its wordcounts. 
// Initialize SparkContext (sc) - assumed already created

// 1. Load text file into RDD
val textRDD = sc.textFile("text.txt")

// 2. Split each line into words (flatMap to get all words)
val wordsRDD = textRDD.flatMap(line => line.split("\\s+"))

// 3. Map each word to a (word, 1) pair
val pairsRDD = wordsRDD.map(word => (word, 1))

// 4. Reduce by key (word) to sum counts
val wordCountsRDD = pairsRDD.reduceByKey(_ + _)

// 5. Collect or print the word counts
wordCountsRDD.collect().foreach{ case (word, count) =>
  println(s"$word : $count")
}


---

Write below queries in Hive;  
1. Write hive query to create databases name: emp. (2 Marks)  
2. Write hive query to CREATE EXTERNAL TABLE in emp name it- employee 
with emp_id,name,location, dep,designation and salary as columns (4 Marks)  
3. Write a hive query to print the average salary of employees based on location. (4 
Marks) 

Here are your Hive queries in one neat window for easy reference:

```sql
-- 1. Create a database named emp
CREATE DATABASE emp;

-- 2. Create an external table named employee in emp database
USE emp;

CREATE EXTERNAL TABLE employee (
  emp_id INT,
  name STRING,
  location STRING,
  dep STRING,
  designation STRING,
  salary FLOAT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/hive/warehouse/emp/employee';

-- 3. Query to get average salary of employees grouped by location
SELECT location, AVG(salary) AS avg_salary
FROM employee
GROUP BY location;
```

---

Write commands / query in MongoDB 
1. Create a collection named users. (2 mark) 
2. Insert below records in users. (2 marks) 
{ name:Abhishek},{name: Raja, age: 24},{name: Ravi,age: 34},{name: Ram, age: 
45},{name: Roopa, age: 44},{name: Tina, age: 54} 
3. Fetch users with age greater than or equal to 34. (2 marks) 
4. Update record {name:Abhishek } with age 34. (2 marks) 
5. Delete the record {name: Ravi }. (2 marks) 

// 1. Create a collection named users
db.createCollection("users")

// 2. Insert records into the users collection
db.users.insertMany([
  { name: "Abhishek" },
  { name: "Raja", age: 24 },
  { name: "Ravi", age: 34 },
  { name: "Ram", age: 45 },
  { name: "Roopa", age: 44 },
  { name: "Tina", age: 54 }
])

// 3. Fetch users with age greater than or equal to 34
db.users.find({ age: { $gte: 34 } })

// 4. Update record {name: "Abhishek"} with age 34
db.users.updateOne(
  { name: "Abhishek" },
  { $set: { age: 34 } }
)

// 5. Delete the record {name: "Ravi"}
db.users.deleteOne({ name: "Ravi" })


Write commands/query in MongoDB to, 
 
i. Create a collection named orders. (1 mark) 
ii. Insert below record in orders. (2 mark) 
{"order_id”: 1, 
"order_date”: '2013-07-25 00:00:00.0', 
"order_customer_id”: 11599,  
"order_status”: "CLOSED" } 
 
iii. Fetch orders with order_status as COMPLETE. (2 marks) 
 
iv. Compute count of orders with status COMPLETE and CLOSED. (2 marks)

Sure! Here's all the MongoDB commands combined into a **single code block**:

```javascript
// i. Create collection 'orders'
db.createCollection("orders")

// ii. Insert the given record
db.orders.insertOne({
  "order_id": 1,
  "order_date": "2013-07-25 00:00:00.0",
  "order_customer_id": 11599,
  "order_status": "CLOSED"
})

// iii. Fetch orders with order_status as "COMPLETE"
db.orders.find({ order_status: "COMPLETE" })

// iv. Count orders with status "COMPLETE" or "CLOSED"
db.orders.countDocuments({ order_status: { $in: ["COMPLETE", "CLOSED"] } })

// Optional: Count separately for each status
db.orders.aggregate([
  { $match: { order_status: { $in: ["COMPLETE", "CLOSED"] } } },
  { $group: { _id: "$order_status", count: { $sum: 1 } } }
])
```




---


 The employee details dataset is provided and loaded as Spark-DataFrame. Using Spark libraries 
execute the steps, as questioned below. 
 
a) 
Using PySpark Dataframe or Spark-SQL libraries process the Dataframe to find out solutions of 
queries mentioned below. 
 
I. What is the average salary of the employees in the dataset? ( 1 marks)   
II. What is the total number of years of experience for all employees? (2 Marks) 
III. What is the gender distribution in the DataFrame? ( 3 marks) 
IV. What is the salary of the employee with the maximum experience? ( 3 marks) 
V. How many employees are older than 22 years? (3 marks) 
VI. Remove the feature 'ID' and also remove null values from the DataFrame. (3 marks) 


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# 1. Create SparkSession (if not already created)
spark = SparkSession.builder \
    .appName("EmployeeDataAnalysis") \
    .getOrCreate()

# 2. Load CSV file into DataFrame
df = spark.read.csv("path/to/employee.csv", header=True, inferSchema=True)

# 3. Show the first few rows to verify data
print("Data Preview:")
df.show(5)

# 4. I. Average salary of employees
print("Average Salary:")
df.selectExpr("avg(salary) as average_salary").show()

# 5. II. Total years of experience for all employees
print("Total Years of Experience:")
df.selectExpr("sum(experience) as total_experience").show()

# 6. III. Gender distribution in the DataFrame
print("Gender Distribution:")
df.groupBy("gender").count().show()

# 7. IV. Salary of employee with maximum experience
print("Employee with Maximum Experience and their Salary:")
df.orderBy(col("experience").desc()).select("name", "experience", "salary").limit(1).show()

# 8. V. Number of employees older than 22 years
print("Number of Employees Older Than 22 Years:")
df.filter(col("age") > 22).select(count("*").alias("employees_above_22")).show()

# 9. VI. Remove 'ID' column and drop rows with null values
df_cleaned = df.drop("ID").na.drop()

print("Data after removing 'ID' column and dropping nulls:")
df_cleaned.show(5)

# 10. Stop SparkSession when done (optional)
# spark.stop()



Using PySpark ML build a regression model, as questioned below. 
 
I. Convert all string columns into numeric values using StringIndexer transformer and 
make sure now DataFrame does not have any string columns anymore. (5 marks)  
II. Using vectorAssembler combines all columns (except target column i.e., 'salary') of 
spark DataFrame into a single column (name as features). Make sure DataFrame now 
contains only two columns: features and salary. (6 marks) 
III. Split the vectorized dataframe into training and test sets with one fourth records being 
held for testing (4 marks) 
IV. Build a LinerRegression model on train set use featuresCol="features" and 'salary'(5 
marks) 
V. Perform prediction on the testing data and Print RMSE,MAE value? (5 marks) 

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Assume your DataFrame is 'df' and target column is 'salary'

# --- I. Convert all string columns to numeric using StringIndexer ---
string_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() == 'string']

for col_name in string_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed")
    df = indexer.fit(df).transform(df)

# Drop original string cols and rename indexed cols to original names
for col_name in string_cols:
    df = df.drop(col_name).withColumnRenamed(col_name + "_indexed", col_name)

# Confirm no string columns remain
df.printSchema()

# --- II. VectorAssembler to combine features into single column 'features' ---
feature_cols = [col for col in df.columns if col != 'salary']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vector = assembler.transform(df).select("features", "salary")

df_vector.show(5)

# --- III. Split into training (75%) and test (25%) sets ---
train_data, test_data = df_vector.randomSplit([0.75, 0.25], seed=42)

# --- IV. Build Linear Regression model on train set ---
lr = LinearRegression(featuresCol="features", labelCol="salary")
lr_model = lr.fit(train_data)

# --- V. Predict on test set and evaluate RMSE and MAE ---
predictions = lr_model.transform(test_data)

evaluator_rmse = RegressionEvaluator(labelCol="salary", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="salary", predictionCol="prediction", metricName="mae")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")




Bangalore Housing Dataset is provided and loaded as Spark-DataFrame.  Using 
Spark libraries execute the steps, as questioned below. 
 
a) 
Using PySpark and Spark-SQL libraries process the dataset to find out solutions 
of queries mentioned below. 
 
i. Count the total number of housing-properties listed from 'HSR Layout' 
location. (2 marks) 
ii. How many ‘2 BHK’ size housing-properties are listed from 'Whitefield' 
location? (3 marks) 
iii. What is the average price of ‘2 BHK’ size housing-properties in ‘HSR 
Layout’ location? (5 Marks) 


```python
# Assume Spark DataFrame `df` with columns: location, size, price

# Create temp view for Spark SQL
df.createOrReplaceTempView("housing")

# i. Count total properties from 'HSR Layout' using DataFrame API and SQL
hsr_count = df.filter(df.location == "HSR Layout").count()
print(f"Total properties in HSR Layout: {hsr_count}")

hsr_count_sql = spark.sql("""
    SELECT COUNT(*) AS total_properties 
    FROM housing 
    WHERE location = 'HSR Layout'
""")
hsr_count_sql.show()

# ii. Count '2 BHK' properties from 'Whitefield' using DataFrame API and SQL
whitefield_2bhk_count = df.filter((df.location == "Whitefield") & (df.size == "2 BHK")).count()
print(f"2 BHK properties in Whitefield: {whitefield_2bhk_count}")

whitefield_2bhk_sql = spark.sql("""
    SELECT COUNT(*) AS total_2bhk_whitefield
    FROM housing
    WHERE location = 'Whitefield' AND size = '2 BHK'
""")
whitefield_2bhk_sql.show()

# iii. Average price of '2 BHK' in 'HSR Layout' using DataFrame API and SQL
avg_price_2bhk_hsr = df.filter((df.location == "HSR Layout") & (df.size == "2 BHK")) \
                      .agg({"price": "avg"}).collect()[0][0]
print(f"Average price of 2 BHK in HSR Layout: {avg_price_2bhk_hsr}")

avg_price_2bhk_hsr_sql = spark.sql("""
    SELECT AVG(price) AS avg_price_2bhk_hsr
    FROM housing
    WHERE location = 'HSR Layout' AND size = '2 BHK'
""")
avg_price_2bhk_hsr_sql.show()
```
