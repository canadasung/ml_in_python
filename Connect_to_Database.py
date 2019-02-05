# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

####################################
# Using Python to connect Database #
####################################

# Prepare Basic Login Information
# the default hostname is 'localhost'
hostname = 'localhost' 
username = 'username'
password = 'password'
database = 'dbname'

# Simple routine to run a query on a database and print the results:
def doQuery( conn ) :
    cur = conn.cursor()

    cur.execute("""SELECT * FROM public.table WHERE date >= '2018-12-01' limit 100;""")

    for user_id, payer in cur.fetchall() :
        print(user_id, payer)

# Connecting to Postgresql Database
# Pull data from PostgresDatabase
# Run the following pip install command in command prompt to get the library "pip install psycopg2" (without quotes)
print ("Using psycopg2…")
import psycopg2
myConnection = psycopg2.connect( host=hostname, user=username, password=password, dbname=database )

# Check if SQL data pulling function normally
#doQuery( myConnection )

cur = myConnection.cursor()
#d1 = cur.execute("""SELECT * FROM public.table WHERE date >= '2018-12-01' limit 100;""")
#df = pd.DataFrame(d1.fetchall())
#df.columns = d1.keys()

# Prepare the query
# Method One using one double quote with back slash
sql_query = "select * \
            from public.table \
            order by date desc \
            limit 100;"
# Method Two using triple double quotes
sql_query = """select * 
            from public.table
            order by date desc
            limit 100;"""


# Write the query result to pandas dataframe
df = pd.read_sql(sql_query, myConnection)

# Close the connection
myConnection.close()



# Connecting to MSSQL
# Run the following pip install command in command prompt to get the library "pip install pymssql" (without quotes)
import pymssql
import os.path


# Setting up debug flag
global dbug
dbug = 1

# Open database connection
conn = pymssql.connect (host='cypress.csil.sfu.ca', user='s_wnsung', password='2drHy3GbYhTtH7Ry', database='wnsung354')
if (dbug == 1):
    print ("after connect")

# Prepare a cursor object using cursor() method
mycursor = conn.cursor()


######################### Main Function ###########################

# Display all product colors
mycursor.execute('Select DISTINCT Color FROM AdventureWorksLT.SalesLT.Product Where Color is not NULL Order by Color')
row = mycursor.fetchone()
while row:
    print(row[0])
    row = mycursor.fetchone()



# Close connection
conn.close()

