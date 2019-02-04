# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Setup Database Access Infomation
hostname = 'your_hostPath'
username = 'your_userName'
password = 'your_password'
database = 'your_database'

# Pull data from PostgresDatabase
print ("Using psycopg2…")
import psycopg2
myConnection = psycopg2.connect( host=hostname, user=username, password=password, dbname=database )
cur = myConnection.cursor()

# Prepare the Query
sql_query = """you full sql queries here;"""

# Write to Pandas DataFrame From SQL result
new_df = pd.read_sql(sql_query, myConnection)

# Close the connection
myConnection.close()

# Read Data from Local Drive
new_df = pd.read_csv("C:/Users/user_name/Documents/your_file.csv")

# DataFrame Information
new_df.info()
new_df = new_df.sort_values(by = ['column_name'], ascending = False)
new_df.to_csv('user_archive.csv') # write to csv
ids = new_df["user_id"]

# Check Duplicates in Dataframe
new_df[ids.isin(ids[ids.duplicated()])].sort_values("column_name") # method 1
pd.concat(g for _, g in new_df.groupby("column_name") if len(g) > 1) # method 2


# Add a column to dataframe
df = pd.DataFrame(y_kmeans)  # create a dataframe from some result
df.info()
verticalStack = pd.concat([dataset, df], axis = 1)  # merge/concatenate the newly created dataframe to an existing dataframe
verticalStack.rename(columns = {0:'segments'}, inplace = True) #rename the column
print(verticalStack.columns)
verticalStack.info() 

# Export to CSV
verticalStack.to_csv('seg3.csv')


# Using DataFrame.drop
df.drop(df.columns[[1, 2]], axis=1, inplace=True)

# drop by Name
df1 = df1.drop(['B', 'C'], axis=1)

# Select the ones you want
df1 = df[['a','d']]

# Change Pandas DataFrame Display Size
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Counts for each row, counting horizontally with non-null cells
final2['count'] = final2.count(axis = 'columns')

# Remove Duplicate Rows but Keep the First Occurrence
final2.drop_duplicates(subset = "column_name", keep = 'first', inplace = True)

# Assign Day 1 tp Day N Priorities
final2 = final2.assign(column_name = 0)
final2.info()
final2.column_name.iloc[0:50] = 1
final2.column_name.iloc[50:150] = 2
final2.column_name.iloc[150:650] = 3
final2.column_name.iloc[650:1650] = 4
final2.column_name.iloc[1650:6650] = 5
final2.column_name.iloc[6650:16650] = 6
final2.column_name.iloc[16650:36650] = 7
final2.column_name.iloc[36650:76650] = 8

# Remove rows with the given thresh number of na columns
# In this example, a row would be removed if contains 5 or more na cells
final2.dropna(thresh = 5, inplace=True)

