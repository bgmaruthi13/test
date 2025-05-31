# Q1 - Section C - Feb 2025

## from pyspark.sql import SparkSession
# Create Spark session
## spark = SparkSession.builder.appName("PlacementAnalysis").getOrCreate()

mba_place = spark.read.load(
    "dbfs:/FileStore/tables/SparkSQL/mba_placement.csv",
    format="csv",
    sep=",",
    inferSchema=True,
    header=True
)
mba_place.count()
mba_place.printSchema()
mba_place_clean = mba_place.drop("sl_no").dropna()
mba_place_clean.show()
## I.	What’s the overall minimum, maximum and average salary from the dataset? ( 6 marks)  
from pyspark.sql.functions import min, max, avg
mba_place_clean.select(
    min("salary").alias("Min_Salary"),
    max("salary").alias("Max_Salary"),
    avg("salary").alias("Avg_Salary")
).show()
## II.	How many  female candidates are not placed ?  ( 4 marks)
mba_place_clean.filter((mba_place_clean.gender == "F") & (mba_place_clean.status == "Not Placed")).count()
## III.	Out of total male candidates placed, how many do not have any work experience ? (3 marks)
mba_place_clean.filter(
    (mba_place_clean.gender == "M") &
    (mba_place_clean.status == "Placed") &
    (mba_place_clean.workex == "No")
).count()

# Q1 - Section C - Oct 2024
df_employee=spark.read.load("dbfs:/FileStore/tables/SparkSQL/employee_data_2000.csv",
                            format="csv",
                            sep=",",
                            inferSchema="True",
                            header=True
)
df_employee.show()
df_employee.count()
df_employee.printSchema()

#What is the average salary of the employees in the dataset? ( 1 marks)
from pyspark.sql.functions import avg
df_employee.select(avg(df_employee.salary).alias("Average Salary")).show()

# What is the total number of years of experience for all employees? (2 Marks)
from pyspark.sql.functions import sum
df_employee.select(sum(df_employee.experience).alias("total numer of exp")).show()

# What is the gender distribution in the DataFrame? ( 3 marks)
# from pyspark.sql.funtions import count
df_employee.groupBy("gender").count().show()

# What is the salary of the employee with the maximum experience? ( 3 marks)
from pyspark.sql.functions import max
max_exp = df_employee.select(max(df_employee.experience)).collect()[0][0]
df_employee.filter(df_employee.experience==max_exp).select(df_employee.name,df_employee.experience,df_employee.salary).show()

## How many employees are older than 22 years? (3 marks)
df_employee.filter(df_employee.age>22).count()


# Remove the feature 'ID' and also remove null values from the DataFrame. (3 marks)
df_employee_clean=df_employee.drop(df_employee.ID).dropna()
df_employee_clean.printSchema()

## Q paper - July 2024 
univ_rank=spark.read.load("dbfs:/FileStore/tables/SparkSQL/qs_world_university_rankings_3000.csv",
                          format="csv",
                          sep=",",
                          inferSchema=True,
                          header=True)
univ_rank.printSchema()
univ_rank.count()
univ_rank.show()

# How many Institutions are included in the dataset? (2 mark)
univ_rank.select(univ_rank["Institution Name"]).distinct().count()

# How many Institutions from ‘India' are included in dataset? (3 marks)
univ_rank.filter(univ_rank["Location"]=='India').count()
univ_rank.filter(univ_rank["Location"] == "India").select("Institution Name").distinct().count()
univ_rank.filter(univ_rank["Location"] == "India").select("Institution Name").distinct().show()

# Print the average "Citations per Faculty" for universities located in 'India'? (5 marks)
univ_rank.filter(univ_rank["Location"]=='India').select(avg("Citations per Faculty")).show()

from pyspark.sql.functions import avg
univ_rank.filter(univ_rank["Location"] == "India") \
  .select(avg("Citations per Faculty").alias("avg_citations_per_faculty")) \
  .show()


# List Institutions where "International Students" percentage is 100 % along with their
#location ( "Location Full"). (5 marks)
univ_rank.show()
univ_rank.filter(univ_rank["International Students"]==100).select("International Students","Location Full").show()
univ_rank = univ_rank.withColumn("International Students", univ_rank["International Students"].cast("double"))

# Question no 2 - Section C
mba_placement = spark.read.load(
    "dbfs:/FileStore/tables/SparkSQL/mba_placement.csv",
    format="csv",
    sep=",",
    inferSchema=True,
    header=True
)
mba_placement.printSchema()

# I.	Convert all string columns into numeric values using StringIndexer transformer and make sure now DataFrame does not have any string columns anymore. (5 marks)
from pyspark.ml.feature import StringIndexer

# List of string columns from schema
string_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]

# Loop through each string column and apply StringIndexer
for col_name in string_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed")
    mba_placement = indexer.fit(mba_placement).transform(mba_placement).drop(col_name)  # drop original string column

# Rename new indexed columns to original names
for col_name in string_cols:
    mba_placement = mba_placement.withColumnRenamed(col_name + "_indexed", col_name)

# Confirm schema no longer contains string columns
mba_placement.printSchema()

# Using vectorAssembler combines all columns (except target column i.e., 'salary') of spark DataFrame into a single column (name as features). Make sure DataFrame now contains only two columns: features and salary. ( 5 marks)
from pyspark.ml.feature import VectorAssembler

# List all columns except 'salary' (target column)
feature_cols = [col for col in mba_placement.columns if col != 'salary']

# Initialize VectorAssembler
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

# Transform the DataFrame
df_vectorized = assembler.transform(mba_placement)

# Select only the 'features' and 'salary' columns
final_df = df_vectorized.select('features', 'salary')

# Show resulting schema
final_df.printSchema()
final_df.show(5, truncate=False)


from pyspark.sql.functions import col

# Count rows with null in 'features' or 'salary'
null_counts = final_df.filter(
    col("features").isNull() | col("salary").isNull()
).count()

print(f"Number of rows with null values: {null_counts}")


# Drop rows where any column has a null value
clean_df = final_df.dropna()

# Show the cleaned DataFrame
clean_df.show()


from pyspark.sql.functions import col

# Count rows with null in 'features' or 'salary'
null_counts = clean_df.filter(
    col("features").isNull() | col("salary").isNull()
).count()

print(f"Number of rows with null values: {null_counts}")


#III.	Split the vectorized dataframe into training and test sets with one fourth records being held for testing (3marks)
# Split the DataFrame: 75% for training, 25% for testing
train_df, test_df = clean_df.randomSplit([0.75, 0.25], seed=42)

# Optional: Check counts
print("Training Set Count:", train_df.count())
print("Test Set Count:", test_df.count())


#IV.	Build a LinearRegression model on train set  use featuresCol="features" and  'salary'(6 marks)
from pyspark.ml.regression import LinearRegression

# Define the Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="salary")

# Fit the model on training data
lr_model = lr.fit(train_df)

# Print model coefficients and intercept
print("Coefficients:", lr_model.coefficients)
print("Intercept:", lr_model.intercept)


# V.	Perform prediction on the testing data and Print MSE value? ( 6 marks)
from pyspark.ml.evaluation import RegressionEvaluator

# Step 1: Predict salaries on the test set
predictions = lr_model.transform(test_df)

# Step 2: Initialize evaluator for MSE
evaluator = RegressionEvaluator(
    labelCol="salary", 
    predictionCol="prediction", 
    metricName="mse"
)

# Step 3: Compute Mean Squared Error
mse = evaluator.evaluate(predictions)

print(f"Mean Squared Error (MSE) on test data: {mse}")


# Oct 2024 Section C

# In[ ]:


#I.	Convert all string columns into numeric values using StringIndexer transformer and make sure now DataFrame does not have any string columns anymore. (5 marks) 

emp_data = spark.read.load(
    "dbfs:/FileStore/tables/SparkSQL/employee_data_2000.csv",
    format="csv",
    sep=",",
    inferSchema=True,
    header=True
)


emp_data.printSchema()


#I.	Convert all string columns into numeric values using StringIndexer transformer and make sure now DataFrame does not have any string columns anymore. (5 marks) 
from pyspark.ml.feature import StringIndexer

# List of string columns from schema
string_cols = ["name", "gender"]

# Loop through each string column and apply StringIndexer
for col_name in string_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed")
    emp_data = indexer.fit(emp_data).transform(emp_data).drop(col_name)  # drop original string column

# Rename new indexed columns to original names
for col_name in string_cols:
    emp_data = emp_data.withColumnRenamed(col_name + "_indexed", col_name)

# Confirm schema no longer contains string columns
emp_data.printSchema()


emp_data=emp_data.drop("ID")


emp_data.printSchema()


emp_data.show(10)


#II.	Using vectorAssembler combines all columns (except target column i.e., 'salary') of spark DataFrame into a single column (name as features). Make sure DataFrame now contains only two columns: features and salary. (6 marks)

from pyspark.ml.feature import VectorAssembler

# List all columns except 'salary' (target column)
feature_cols = [col for col in emp_data.columns if col != 'salary']

# Initialize VectorAssembler
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

# Transform the DataFrame
df_vectorized = assembler.transform(emp_data)

# Select only the 'features' and 'salary' columns
final_df = df_vectorized.select('features', 'salary')

# Show resulting schema
final_df.printSchema()
final_df.show(5, truncate=False)


from pyspark.sql.functions import col

# Count rows with null in 'features' or 'salary'
null_counts = final_df.filter(
    col("features").isNull() | col("salary").isNull()
).count()

print(f"Number of rows with null values: {null_counts}")


from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaler_model = scaler.fit(final_df)
df_scaled = scaler_model.transform(final_df)


#III.	Split the vectorized dataframe into training and test sets with one fourth records being held for testing (4 marks)

train_df, test_df = df_scaled.randomSplit([0.75, 0.25], seed=42)

# Optional: Check counts
print("Training Set Count:", train_df.count())
print("Test Set Count:", test_df.count())


#IV.	Build a LinerRegression model on train set use featuresCol="features" and 'salary'(5 marks)

from pyspark.ml.regression import LinearRegression

# Define the Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="salary")

# Fit the model on training data
lr_model = lr.fit(train_df)

# Print model coefficients and intercept
print("Coefficients:", lr_model.coefficients)
print("Intercept:", lr_model.intercept)



# V.	V.	Perform prediction on the testing data and Print RMSE,MAE value? (5 marks)
from pyspark.ml.evaluation import RegressionEvaluator

# Step 1: Predict salaries on the test set
predictions = lr_model.transform(test_df)

# Step 2: Initialize evaluator for MSE
evaluator = RegressionEvaluator(
    labelCol="salary", 
    predictionCol="prediction", 
    metricName="mse"
)

# Step 3: Compute Mean Squared Error
mse = evaluator.evaluate(predictions)

print(f"Mean Squared Error (MSE) on test data: {mse}")


# # july 2024 Paper review 
# 
# 

# In[ ]:


# Recreate the Dataframe by Dropping all the rows where 'QS Overall Score' is
# mention as '-' and also convert it as float type. (2 marks )

univ_data = spark.read.load(
    "dbfs:/FileStore/tables/SparkSQL/QS_World_University_Rankings_1200.csv",
    format="csv",
    sep=",",
    inferSchema=True,
    header=True
)


univ_data.printSchema()


univ_data.count()


# Recreate the Dataframe by Dropping all the rows where 'QS Overall Score' is
# mention as '-' and also convert it as float type. (2 marks )

from pyspark.sql.functions import col

# Step 1: Filter out rows where 'QS Overall Score' is '-'
df_cleaned = univ_data.filter(col("QS Overall Score") != "-")

# Step 2: Cast 'QS Overall Score' to float
df_cleaned = df_cleaned.withColumn("QS Overall Score", col("QS Overall Score").cast("float"))

# Step 3: Verify schema and sample data
df_cleaned.printSchema()
df_cleaned.show(5)


df_cleaned.count()


#Remove all the rows with any missing entry.( 3 marks)

df_no_nulls = df_cleaned.dropna()


df_no_nulls.count()


df_no_nulls.printSchema()


from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer

# Step 1: Identify string columns
string_cols = [field.name for field in df_no_nulls.schema.fields if isinstance(field.dataType, StringType)]

# Step 2: Apply StringIndexer to all string columns and drop originals
for col_name in string_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed")
    df_no_nulls = indexer.fit(df_no_nulls).transform(df_no_nulls).drop(col_name)

# Step 3: Rename indexed columns back to original names
for col_name in string_cols:
    df_no_nulls = df_no_nulls.withColumnRenamed(col_name + "_indexed", col_name)

# Final DataFrame
df_indexed = df_no_nulls

# Step 4: Confirm schema has no string columns
df_indexed.printSchema()


'''Using vectorAssembler combines all columns, except target column i.e. 'QS Overall
Score', of spark DataFrame into single column (name it as features). Make sure
DataFrame now contains only two columns, 'features' and 'QS Overall Score'. (5
marks)'''

from pyspark.ml.feature import VectorAssembler

# Exclude the target column
feature_cols = [col for col in df_indexed.columns if col != "QS Overall Score"]

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_us")
df_vector = assembler.transform(df_indexed).select("features_us", "QS Overall Score")


scaler = StandardScaler(inputCol="features_us", outputCol="features", withMean=True, withStd=True)
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)


#Split the vectorised Dataframe into training and test sets with one fifth records being
#held for testing. (2 marks)

train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)


'''Train default LinearRegression model with features as 'featuresCol' and ‘QS Overall
Score’ as label on training set. (3 marks)'''

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="QS Overall Score")
lr_model = lr.fit(train_df)


'''Perform prediction on the testing data and Print RMSE value. (5 marks)'''

from pyspark.ml.evaluation import RegressionEvaluator

# Perform prediction
predictions = lr_model.transform(test_df)

# Evaluate RMSE
evaluator = RegressionEvaluator(
    labelCol="QS Overall Score",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")


# # Section B - Q1 

# Write HDFS shell commands for the following- 
# 1.	 Print contents of the directory  by path, showing the names, permissions, owner, size and modification date for each entry? (2 marks)
# 2.	How do you upload multiple files from the local system to a directory in HDFS?. (2 marks)
# 3.	How do you display the contents of a file in HDFS line by line (paged view)? (2 marks)
# 4.	Write command to remove a file or directory identified by path in HDFS and recursively delete any child entries. (2 marks)
# 5.	Write command to copy the ‘testfile’ of the hadoop filesystem to the local file system  (2marks)
# 
# Note: Consider InputDir, OutputDir, XYZ, SampleDir, and file.txt are under the present working directory.
# 

# In[ ]:


1> 1. Print contents of the directory by path, showing the names, permissions, owner, size and modification date for each entry? 

 hdfs dfs -ls /user/hadoop/InputDir


2>2. How do you upload multiple files from the local system to a directory in HDFS?. (2 marks) 

hdfs dfs -put *.txt /user/hadoop/SampleDir
hdfs dfs -put file1.txt file2.txt file3.txt /path/in/hdfs


3. How do you display the contents of a file in HDFS line by line (paged view)?

hdfs dfs -cat /user/hadoop/file.txt | less


4. Write command to remove a file or directory identified by path in HDFS and recursively delete any child entries. (2 marks)

hdfs dfs -rm -r /user/hadoop/XYZ


5. Write command to copy the ‘testfile’ of the hadoop filesystem to the local file system (2marks)

hdfs dfs -get /user/hadoop/testfile ./testfile


# Write HDFS shell commands for the following-
# 1. To Copy file1.txt from folder InputDir to OutputDir as file2.txt. (2 marks)
# 2. To Delete an empty directory named as XYZ. (2 marks)
# 3. To List the files and directories under folder named SampleDir. (2 marks)
# 4. To Recursively list the files and directories exist under folder named SampleDir. (2 marks)
# 5. To change the Permission of file named file.txt to Read only (444) (2marks)
# Note: Consider InputDir, OutputDir, XYZ, SampleDir, and file.txt are under the present
# working directory.

# In[ ]:


To Copy file1.txt from folder InputDir to OutputDir as file2.txt. (2 marks)

hdfs dfs -cp /user/hadoop/InputDir/file1.txt /user/hadoop/OutputDir/file2.txt


To Delete an empty directory named as XYZ. (2 marks)

hdfs dfs -rmdir /user/hadoop/XYZ


To List the files and directories under folder named SampleDir. (2 marks)

hdfs dfs -ls /user/hadoop/SampleDir


To Recursively list the files and directories exist under folder named SampleDir. (2 marks)

hdfs dfs -ls -R /user/hadoop/SampleDir



Change the permission of file named file.txt to read-only (444)

hdfs dfs -chmod 444 /user/hadoop/file.txt


# 2022 paper 

# In[ ]:


To print the version of installed Hadoop (2 marks)

hadoop version


To copy file1.txt from folder InputDir to OutputDir as file2.txt (3 marks)

hdfs dfs -cp /user/hadoop/InputDir/file1.txt /user/hadoop/OutputDir/file2.txt


To delete an empty directory named XYZ (3 marks)

hdfs dfs -rmdir /user/hadoop/XYZ


To list the contents of a folder named SampleDir (3 mark
                                                  
 hdfs dfs -ls /user/hadoop/SampleDir                                                 


To fetch the usage instructions/details of the mkdir command (3 marks)

hdfs dfs -help mkdir


# Section B - Question B Rdd

# In[ ]:


Considering sc as spark content object, and rdd as RDD object, write Spark commands to,
1.	Create an RDD from the following list: List(1, 2, 3, 4, 5,6,7,8,9,10). (2 marks)
2.	Display/Print first four elements of the RDD. (2 marks)
3.	Display/Print the first element of the RDD. (2 marks)
4.	Explain with example map,filter Apache Spark transformations.(4 marks)


#1.	Create an RDD from the following list: List(1, 2, 3, 4, 5,6,7,8,9,10). (2 marks)

rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# Display/Print first four elements of the RDD. (2 marks)

print(rdd.take(4))


#3.	Display/Print the first element of the RDD. (2 marks)

print(rdd.first())


# Explain with example map,filter Apache Spark transformations.(4 marks)

# Definition: map is used to transform each element of the RDD.

rdd_mapped = rdd.map(lambda x: x * 2)
print(rdd_mapped.collect())


# Definition: filter is used to retain only those elements that satisfy a condition.


rdd_filtered = rdd.filter(lambda x: x % 3 == 0)
print(rdd_filtered.collect())


Considering sc as spark content object, and rdd as RDD object, write Spark commands to,
1.	Create an RDD from the following list: List(1, 2, 3, 4, 5,6). (2 marks)
2.	Display/Print first four elements of the RDD. (2 marks)
3.	Display/Print the first element of the RDD. (2 marks)
4.	Display/Print the number of elements in the RDD. (2 marks)
5.	Display sum of all elements of the RDD. (2 marks)


#1.	Create an RDD from the following list: List(1, 2, 3, 4, 5,6). (2 marks)
rdd=sc.parallelize([1, 2, 3, 4, 5,6])


#2.	Display/Print first four elements of the RDD. (2 marks)

print(rdd.take(4))


# 3.	Display/Print the first element of the RDD. (2 marks)

print(rdd.first())


# Display/Print the number of elements in the RDD. (2 marks)

print(rdd.count())


# Display sum of all elements of the RDD. (2 marks)
print(rdd.sum())


Considering sc as spark content object, and rdd as RDD object, write Spark commands to,
1. Create an RDD from the following list: List(1, 2, 3, 4, 5,6). (2 marks)
2. Read/load a text file located at "/path/to/file.txt" into an RDD. (2 marks)
3. Filter out the even numbers from RDD. (2 marks)
4. Map each element in the RDD to its square. (2 marks)
5. Count the number of elements in the RDD. (2 marks)


# . Create an RDD from the following list: List(1, 2, 3, 4, 5,6). (2 marks)
rdd=sc.parallelize([1, 2, 3, 4, 5,6])


# 2. Read/load a text file located at "/path/to/file.txt" into an RDD. (2 marks)
text_rdd = sc.textFile("/path/to/file.txt")


# 3. Filter out the even numbers from RDD. (2 marks)

odd_rdd = rdd.filter(lambda x: x % 2 != 0)
print(odd_rdd.collect())


# 4. Map each element in the RDD to its square. (2 marks)
sq_rdd=rdd.map(lambda x:x**2)
print(sq_rdd.collect())


# 5. Count the number of elements in the RDD. (2 marks)
print(rdd.count())


Write below queries in Hive; 
1.	Write hive query to create databases name: emp. (2 Marks) 
2.	Write hive query to CREATE EXTERNAL TABLE in emp name it- employee with emp_id,name,location, dep,designation and salary as columns (4 Marks) 
3.	Write a hive query to perform an inner join on the Table1 and Table 2 on ‘id’ column (4 Marks) 


#Write hive query to create databases name: emp. (2 Marks) 

get_ipython().run_line_magic('sql', '')
CREATE DATABASE emp;


#Write hive query to CREATE EXTERNAL TABLE in emp name it- employee with emp_id,name,location, dep,designation and salary as columns (4 Marks) 
get_ipython().run_line_magic('sql', '')
CREATE EXTERNAL TABLE employee (
    emp_id INT,
    name STRING,
    location STRING,
    dep STRING,
    designation STRING,
    salary FLOAT
)
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY '\t' 
LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '/user/hive/warehouse/employee/'
TBLPROPERTIES ("skip.header.line.count"="1");


#Write a hive query to perform an inner join on the Table1 and Table 2 on ‘id’ column (4 Marks) 
get_ipython().run_line_magic('sql', '')
SELECT *
FROM Table1 t1
JOIN Table2 t2
ON t1.id = t2.id;


Write below queries in Hive; 
1.	Write hive query to create databases name: emp. (2 Marks) 
2.	Write hive query to CREATE EXTERNAL TABLE in emp name it- employee with emp_id,name,location, dep,designation and salary as columns (4 Marks) 
3.	Write a hive query to print the average salary of employees based on location. (4 Marks) 


#1.	Write hive query to create databases name: emp. (2 Marks) 
get_ipython().run_line_magic('sql', '')
CREATE DATABASE emp;


#2.	Write hive query to CREATE EXTERNAL TABLE in emp name it- employee with emp_id,name,location, dep,designation and salary as columns (4 Marks)
get_ipython().run_line_magic('sql', '')
CREATE EXTERNAL TABLE emp.employee (
    emp_id INT,
    name STRING,
    location STRING,
    dep STRING,
    designation STRING,
    salary FLOAT
)
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY '\t' 
LINES TERMINATED BY '\n'
TBLPROPERTIES ("skip.header.line.count"="1");


# 3.	Write a hive query to print the average salary of employees based on location. (4 Marks) 
get_ipython().run_line_magic('sql', '')
SELECT location, AVG(salary) AS average_salary
FROM emp.employee
GROUP BY location;


1. Write hive query to create databases name: anotherDB. (2 Marks)
2. Write hive query to CREATE EXTERNAL TABLE in anotherDB name it- orders1
with order_id, order_date, order_customer_id and order_status as columns(4
Marks)
3. Write hive query to load data in orders1 table using file which is available in local file
system. (4 Marks)


#1. Write hive query to create databases name: anotherDB. (2 Marks)
get_ipython().run_line_magic('sql', '')
CREATE DATABASE anotherDB;


#2. Write hive query to CREATE EXTERNAL TABLE in anotherDB name it- orders1
#with order_id, order_date, order_customer_id and order_status as columns(4
#Marks)

CREATE EXTERNAL TABLE anotherDB.orders1 (
    order_id INT,
    order_date STRING,
    order_customer_id INT,
    order_status STRING
)
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n'
TBLPROPERTIES ("skip.header.line.count"="1");


#3. Write hive query to load data in orders1 table using file which is available in local file
#system. (4 Marks)


LOAD DATA LOCAL INPATH '/path/to/your/file.csv' 
INTO TABLE anotherDB.orders1;

# if we need to load data from HDFS 

LOAD DATA INPATH '/path/in/hdfs/orders1.csv' 
INTO TABLE anotherDB.orders1;


Section B - Question d 




Write commands / query in MongoDB
1. Create a collection named ‘product collection’. (2 mark)
2. Insert 5 documents in product collection based on name, rating, brand.(2 mark)
3. Write query to find those products which have received 5/5 rating.(3 mark)
4. Write a query to update those records where the product name is AC to "Air conditioner"and print it. (3 mark)


#1. Create a collection named ‘product collection’. (2 mark)

db.createCollection("product_collection")


#. Insert 5 documents in product collection based on name, rating, brand.(2 mark)
db.product_collection.insertMany([
  { name: "AC", rating: 5, brand: "Samsung" },
  { name: "TV", rating: 4, brand: "Sony" },
  { name: "Washing Machine", rating: 5, brand: "LG" },
  { name: "Microwave", rating: 3, brand: "Panasonic" },
  { name: "Refrigerator", rating: 5, brand: "Whirlpool" }
])


# 3. Write query to find those products which have received 5/5 rating.(3 mark)

db.product_collection.find({ rating: 5 })


#4. Write a query to update those records where the product name is AC to "Air conditioner"and print it. (3 mark)


db.product_collection.updateMany(
  { name: "AC" },
  { $set: { name: "Air conditioner" } }
)

# Print updated documents
db.product_collection.find({ name: "Air conditioner" })


Write commands / query in MongoDB
1. Create a collection named users. (2 mark)
2. Insert below records in users. (2 marks)
{ name:Abhishek},{name: Raja, age: 24},{name: Ravi,age: 34},{name: Ram, age: 45},{name: Roopa, age: 44},{name: Tina, age: 54}
3. Fetch users with age greater than or equal to 34. (2 marks)
4. Update record {name:Abhishek } with age 34. (2 marks)
5. Delete the record {name: Ravi }. (2 marks)


# 1. Create a collection named users. (2 mark)
db.createCollection("users")


#2. Insert below records in users. (2 marks)

db.users.insertMany([
  { name: "Abhishek" },
  { name: "Raja", age: 24 },
  { name: "Ravi", age: 34 },
  { name: "Ram", age: 45 },
  { name: "Roopa", age: 44 },
  { name: "Tina", age: 54 }
])


# Fetch users with age ≥ 34:


db.users.find({ age: { $gte: 34 } })


# 4. Update the record { name: "Abhishek" } by adding age: 34:

db.users.updateOne(
  { name: "Abhishek" },
  { $set: { age: 34 } }
)


# Delete the record { name: "Ravi" }:

db.users.deleteOne({ name: "Ravi" })


# Write commands/query in MongoDB to,
# 1. Create a collection named orders. (2 mark)
# 2. Insert below two records in orders. (4 mark)
# {"order_id”: 1,
# "order_customer_id”: 11599,
# "order_status”: "CLOSED" }
# {"order_id”: 2,
# "order_customer_id”: 11698,
# "order_status”: "OPEN" }
# 3. Fetch orders with order_status as COMPLETE. (4 marks)

# In[ ]:


# Create a collection named orders. (2 mark)
db.createCollection("orders")


#Insert below two records in orders. (4 mark) {"order_id”: 1, "order_customer_id”: 11599, "order_status”: "CLOSED" } {"order_id”: 2, "order_customer_id”: 11698, "order_status”: "OPEN" }
db.orders.insertMany([
  { order_id: 1, order_customer_id: 11599, order_status: "CLOSED" },
  { order_id: 2, order_customer_id: 11698, order_status: "OPEN" }
])


#Fetch orders with order_status as COMPLETE. (4 marks)
db.orders.find({ order_status: "COMPLETE" })
