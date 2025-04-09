#  CMPS664_Project

## This repository contains:
```
- project.py
- online_retail_dataset.csv: CSV file containing the relational dataset
```
  ###  Steps after cloning repository:
  
  1. Open ```project.py``` and ```online_retail_dataset.csv``` in VsCode.
  2. Search for ```mysql.connector.connect()``` function call and modify the following parameters:  
    - host, user and password to allow connection to MySQL server
  3. The default dataset is ```online_retail_dataset.csv``` (already included in the project). Additionally, these are the default functional dependencies and primary keys that should be entered at the command line:
       
     **Functional Dependencies**: {StockCode}->{Description,UnitPrice},{CustomerID}->{Country},{InvoiceNo}->{InvoiceDate,CustomerID},{InvoiceNo,StockCode}->{Quantity}  
     **Primary_keys** = InvoiceNo,StockCode
  5. To change the dataset:  
    a. Download dataset as a CSV file and store it in the same project folder as ```project.py```.  
    b. Search the ```pd.read_csv()``` function and modify the parameter with the name of your CSV data file (located right below the ```def_main()```). 
  6. After running ```project.py```, the interactive query interface will prompt you to enter a SQL query to:  
    a. Insert, update and delete rows within each table  
    b. Compose sql queries for added table insights    
 
