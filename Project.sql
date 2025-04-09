-- drop database if exists project_database;
USE project_database;
SHOW TABLES;

select * from table_1; # this is the product table
select * from table_2; # this is the customer table
select * from table_3; # this is the invoice table -> should be further broken down!!!
select * from table_4; # this is the invoicedetails table 

# Insert Example
INSERT INTO Table_1 (StockCode, Description, UnitPrice)  
VALUES ('A123', 'Wireless Mouse', 25.99);

# Update Example
UPDATE Table_1  
SET UnitPrice = 29.99  
WHERE StockCode = 'A123';

# Delete Example
DELETE FROM Table_4  
WHERE InvoiceNo = 'INV1001';


# 1) What is the total revenue by country?
SELECT 
    t2.Country, 
    SUM(t1.UnitPrice * t4.Quantity) AS TotalRevenue
FROM Table_4 t4
JOIN Table_1 t1 ON t4.StockCode = t1.StockCode
JOIN Table_3 t3 ON t4.InvoiceNo = t3.InvoiceNo
JOIN Table_2 t2 ON t3.CustomerID = t2.CustomerID
GROUP BY t2.Country
ORDER BY TotalRevenue DESC;

# 2) What are the top 5 best selling products? 
SELECT 
    t1.Description, 
    SUM(t4.Quantity) AS TotalQuantitySold
FROM Table_4 t4
JOIN Table_1 t1 ON t4.StockCode = t1.StockCode
GROUP BY t1.Description
ORDER BY TotalQuantitySold DESC
LIMIT 5;

# 3) Average cost that each customer bought
SELECT 
    t3.CustomerID, 
    AVG(t1.UnitPrice * t4.Quantity) AS AvgOrderValue
FROM Table_4 t4
JOIN Table_1 t1 ON t4.StockCode = t1.StockCode
JOIN Table_3 t3 ON t4.InvoiceNo = t3.InvoiceNo
GROUP BY t3.CustomerID
ORDER BY AvgOrderValue DESC;






