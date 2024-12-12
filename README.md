San Francisco Crime Data Analysis
-
Steps to run
-
1. Setup GCP with 1 Master node and 2 Worker nodes
2. Open Jupyter notebook in the GCP Cluster
3. Upload the dataset from <a href="https://www.kaggle.com/code/serkanp/san-francisco-crime-analysis-data-visualization/input" target="_blank">Kaggle</a> to GCS Bucket
4. Create a PySpark Session
5. Read the dataset from jupyter by executing the following command 
```
gcs_file_path = "gs://<Bucket_Name>/<Dataset>.csv"
df = spark.read.csv(gcs_file_path, header=True, inferSchema=True)
```
6. Run the "Analysis.ipynb" in file to view the analysis and graphs. The photos will be stored in the GCS Bucket.
7. Run the "model.ipynb" file to view the prediction model. The prediction model will be stored in the GCS Bucket.

Top 20 Crime Categories in San Francisco
-
![Alt text](OutputImages/Top_20_Crime_Categories.png)

Crime by District
-
![Alt text](OutputImages/Crime_by_District.png)

Crimes by Day of the Week 
-
![Alt text](OutputImages/Crime_by_Day_of_the_Week.png)

Square Graph of Top Crime Categories
-
![Alt text](OutputImages/Square_Graph_of_Top_Crime_Categories.png)

Geo Mapping of Crime on San Francisco Map
-
![Alt text](OutputImages/Geo_Mapping_of_Crime.png)

