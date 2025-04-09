'''
Olivia Arva
CMPS 664
Project #1
'''
# Source: https://archive.ics.uci.edu/dataset/352/online+retail

# Importing packages and libraries
import mysql.connector
import pandas as pd
import re
from itertools import combinations
from io import StringIO


# Function to process the functional dependencies
def break_down_fds(fd_input):
    cleaned_fds = []
    # Split the input string by comma but not within the curly braces
    fds = re.split(r',\s*(?![^{}]*\})', fd_input)
    for fd in fds:
        # Strip leading and trailing spaces
        fd = fd.strip()
        if '->' in fd:
            # Split the functional dependency into left and right parts
            left_fd, right_fd = fd.split('->')
            left_fd = left_fd.strip('{}')
            right_fd = right_fd.strip('{}')

            # Split the left and right parts by comma
            left_fd = {attr.strip() for attr in left_fd.split(',')}
            right_fd = {attr.strip() for attr in right_fd.split(',')}

            # Remove duplicates
            cleaned_fds.append((left_fd, right_fd))
    return cleaned_fds


# Function to compute the closure of a set of attributes
def compute_closure(left_fd, fds):
    closure = set(left_fd)
    # Iterating until no new elements are added to the closure
    while True:
        new_elements = set()
        # Iterate through the functional dependencies
        for fd_left, fd_right in fds:
            if isinstance(fd_left, str):
                fd_left = {fd_left}
            if isinstance(fd_right, str):
                fd_right = {fd_right}
            if set(fd_left).issubset(closure):  # Check if fd_left is a subset of the closure
                new_elements.update(fd_right)
        if new_elements.issubset(closure):  # If no new elements are added, stop
            break
        closure.update(new_elements)  # Add new elements to the closure
    return closure


# Function to detect partial dependencies
def detect_pds(fds, primary_keys):
    partial_dependencies = []
    primary_keys_set = set(primary_keys.strip().split(','))  # Convert primary keys to a set

    for left_fd, right_fd in fds:
        if isinstance(left_fd, str):
            left_fd = {left_fd}
        if isinstance(right_fd, str):
            right_fd = {right_fd}

        # Check if left_fd is a proper subset of the primary key
        if left_fd.issubset(primary_keys_set) and not left_fd == primary_keys_set:
            partial_dependencies.append((left_fd, right_fd))

    return partial_dependencies


# Function to detect transitive dependencies
def detect_tds(fds, primary_keys):
    transitive_dependencies = []
    primary_keys_set = set(primary_keys.strip().split(','))  # Convert primary keys to a set

    for left_fd, right_fd in fds:
        # Ensure left_fd and right_fd are sets
        if isinstance(left_fd, str):
            left_fd = {left_fd}
        if isinstance(right_fd, str):
            right_fd = {right_fd}

        # Check if both left_fd and right_fd are not subsets of the primary keys
        if not left_fd.issubset(primary_keys_set) and not right_fd.issubset(primary_keys_set):
            transitive_dependencies.append((left_fd, right_fd))

    return transitive_dependencies


# Function to suggest candidate keys
def suggest_candidate_keys(df, fds):
    attributes = list(df.columns.str.strip())
    all_candidate_keys = []

    # Generate all combinations of attributes
    for r in range(1, len(attributes) + 1):
        for combo in combinations(attributes, r):
            closure = compute_closure(set(combo), fds)
            if set(closure) == set(attributes):
                # Check minimality
                is_minimal = True
                # Check if any proper subset of the combination is also a candidate key
                for smaller_combo in combinations(combo, r - 1):
                    if compute_closure(set(smaller_combo), fds) == set(attributes):
                        is_minimal = False
                        break
                if is_minimal:
                    all_candidate_keys.append(combo)

    return all_candidate_keys


# 1NF: check if there are multiple values in a single column
def check_1nf(df):
    for column in df.columns:
        # Check if the column contains any multi-valued attributes
        for value in df[column]:
            if isinstance(value, str) and ',' in value:
                print(f"Column '{column}' violates 1NF with this value: {value}")
                return False

    return True


def normalize_to_1nf(df):
    normalized_df = df.copy()

    for column in normalized_df.columns:
        if normalized_df[column].apply(lambda x: isinstance(x, str) and ',' in x).any():
            # Split the multi-valued attributes into separate rows
            normalized_df = normalized_df.drop(column, axis=1).join(
                normalized_df[column].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename(column)
            )

    return normalized_df


# 2NF: check if the non-key attributes are dependent on all primary keys, if not then split the dataset
def check_2nf(df, primary_keys, fds):
    if not check_1nf(df):
        return False
    partial_deps = detect_pds(fds, primary_keys)
    if partial_deps:
        print('Dataset violates 2NF due to these partial dependencies:')
        for left_fd, right_fd in partial_deps:
            print(f"{left_fd} -> {right_fd}")
        return False

    return True


def normalize_to_2nf(df, fds, primary_keys):
    partial_dependencies = detect_pds(fds, primary_keys)
    if not partial_dependencies:
        return [df]  # Return the original table as a list
    new_tables = []

    # Original table without the partial dependencies
    original_table = df.copy()
    for left_fd, right_fd in partial_dependencies:

        # Creating a new table for the dependent attributes
        table_attributes = set(left_fd).union(right_fd)
        new_table = df[list(table_attributes)].drop_duplicates()
        new_tables.append(new_table)
        # print(f"Created new table with attributes: {table_attributes}")
        original_table = original_table.drop(columns=list(right_fd), errors='ignore')

    return [original_table] + new_tables


# 3NF: check if a non-key attribute is dependent on another key attribute, if so then split the dataset
def check_3nf(df_list, primary_keys, fds):
    """
    Check if each table in the list of DataFrames is 3NF compliant.
    """
    all_compliant = True
    for i, df in enumerate(df_list):
        transitive_deps = detect_tds(fds, primary_keys)
        # Checking if there are any transitive dependencies
        if transitive_deps:
            for left_fd, right_fd in transitive_deps:
                print(f"{left_fd} -> {right_fd}")
            all_compliant = False
    return all_compliant


def normalize_to_3nf(df_list, fds, primary_keys):
    """
    Normalize each table in the list of DataFrames to 3NF.
    """
    normalized_tables = []
    for i, df in enumerate(df_list):
        transitive_dependencies = detect_tds(fds, primary_keys)
        if not transitive_dependencies:
            normalized_tables.append(df)
            continue

        new_tables = []
        original_table = df.copy()
        for left_fd, right_fd in transitive_dependencies:

            # Ensure left_fd and right_fd are sets
            if isinstance(left_fd, str):
                left_fd = {left_fd}
            if isinstance(right_fd, str):
                right_fd = {right_fd}

            # Creating a new table for the dependent attributes
            table_attributes = set(left_fd).union(right_fd)
            valid_attributes = [attr for attr in table_attributes if attr in df.columns]
            if not valid_attributes:
                continue

            # Create a new table with both left_fd and right_fd attributes
            new_table = df[valid_attributes].drop_duplicates()
            new_tables.append(new_table)

            # Remove dependent attributes from the original table
            original_table = original_table.drop(columns=list(right_fd), errors="ignore")

        # Add the remaining table and the new tables to the normalized list
        normalized_tables.append(original_table)
        normalized_tables.extend(new_tables)

    return normalized_tables


# BCNF: every  attribute in a table should depend on the key, the whole key, and nothing but the key 
def check_bcnf(df, fds):
    bcnf_violations = []
    attributes = set(df.columns)
    for left_fd, right_fd in fds:
        if isinstance(left_fd, str):
            left_fd = {left_fd}
        if isinstance(right_fd, str):
            right_fd = {right_fd}
        # Check if the determinant (left_fd) is a superkey
        closure = compute_closure(left_fd, fds)
        if not closure.issuperset(attributes):
            bcnf_violations.append((left_fd, right_fd))
    if bcnf_violations:
        # print("Dataset violates BCNF due to the following dependencies:")
        return False, bcnf_violations
    print("The dataset complies with BCNF.")
    return True, []


# After 1NF, 2NF and 3NF normalized tables, checking if bcnf and normalizing if not
def normalize_to_bcnf(df, fds):
    table_counter = 1
    normalized_tables = {}

    # Call check_bcnf once and reuse its result
    is_bcnf, bcnf_violations = check_bcnf(df, fds)
    if is_bcnf:
        print("The dataset is already BCNF compliant.")
        # Add the original table as a single normalized table
        normalized_tables[f"Table_{table_counter}"] = {
            "columns": {col: "VARCHAR(255)" for col in df.columns},
            "data": df.values.tolist()
        }
        return normalized_tables

    print("Dataset violates BCNF due to the following dependencies:")
    for left_fd, right_fd in bcnf_violations:
        print(f"{left_fd} -> {right_fd}")
    print(" ")
    print("Normalizing dataset to BCNF:")
    print(" ")
    for left_fd, right_fd in bcnf_violations:
        # Create a new table for the violating dependency
        columns = list(left_fd.union(right_fd))
        new_table_name = f"Table_{table_counter}"
        table_counter += 1

        # Add the new table to the normalized tables dictionary
        normalized_tables[new_table_name] = {
            "columns": {col: "VARCHAR(255)" for col in columns},
            "data": df[columns].drop_duplicates().values.tolist()
        }
        print(f"Created new table: {new_table_name} with columns: {columns}")

        # Drop the dependent attributes from the original table
        df = df.drop(columns=list(right_fd)).drop_duplicates()

    # Adding the remaining table to the list of normalized tables
    remaining_table_name = f"Table_{table_counter}"
    normalized_tables[remaining_table_name] = {
        "columns": {col: "VARCHAR(255)" for col in df.columns},
        "data": df.values.tolist()
    }
    print(f"Created new table: {remaining_table_name} with columns: {list(df.columns)}")

    return normalized_tables


# Creating the normalized tables
def create_and_populate_tables(mycursor, normalized_tables):
    """
    Create and populate MySQL tables based on the normalized tables.
    """
    for table_name, table_data in normalized_tables.items():
        # Extract columns and data
        columns = table_data["columns"]
        data = table_data["data"]

        # Create the table schema
        column_definitions = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
        create_table_query = f"CREATE TABLE {table_name} ({column_definitions});"
        # print(f"Executing query: {create_table_query}")
        mycursor.execute(f"DROP TABLE IF EXISTS {table_name};")  # Drop table if it already exists
        mycursor.execute(create_table_query)

        # Insert data into the table
        if data:
            placeholders = ", ".join(["%s"] * len(columns))
            insert_query = f"INSERT INTO {table_name} ({', '.join(columns.keys())}) VALUES ({placeholders});"
            # print(f"Executing query: {insert_query} with data: {data}")
            mycursor.executemany(insert_query, data)

    print("All tables have been created and populated successfully.")


def main():
    df = pd.read_csv('online_retail_dataset.csv')
    
    print("=============== Key Information ===============")
    print(f'Num of rows and columns in dataset: {df.shape[0]} , {df.shape[1]}')

    print("Subset sample of dataset")
    print(df.head(5))

    print("Attribute data types of dataset features")
    print(df.dtypes)
    print("  ")

    # Asking user for functional dependencies and primary keys 
    fd_input = input("Specify the functional dependecies (A->B,C->D):")
    primary_keys = input("Specify the primary keys:")
    # fd_input = "{StockCode}->{Description,UnitPrice},{CustomerID}->{Country},{InvoiceNo}->{InvoiceDate,CustomerID},{InvoiceNo,StockCode}->{Quantity}"
    # primary_keys = "InvoiceNo,StockCode"

    # fd_input = "{Student_ID}->{Student_Name},{Student_ID}->{Course, Instructor},{Instructor}->{Instructor_Phone}"
    # primary_keys = "Student_ID, Course"

    print("============ Functional Dependencies ============")
    print(fd_input)
    print("  ")

    print("================= Primary Keys =================")
    print(primary_keys)
    print("   ")

    # Calling 'break_down_function' to correctly parse through the given fds
    cleaned_fds_input = break_down_fds(fd_input)

    print("======== Computing Closure of Attributes ========")
    # Computing closure of attributes only utilized within in the functional dependencies
    for left_fd, _ in cleaned_fds_input:
        print(f"Closure of {left_fd}+: {compute_closure(left_fd, cleaned_fds_input)}")
    print("   ")

    print("============ Potential Candidate Keys ============")
    print(f"Suggested Candidate Keys: {suggest_candidate_keys(df, cleaned_fds_input)}")
    print("   ")

    print("========== Is the table 1NF compliant? ==========")
    is_1nf_compliant = check_1nf(df)

    if not is_1nf_compliant:
        print("1NF Normalized Table ")
        df_1nf = normalize_to_1nf(df)
        print(df_1nf.head(5))
    else:
        print("The dataset is already in 1NF.")
        df_1nf = df.copy()
    print("   ")

    print("========== Is the table 2NF compliant? ==========")
    is_2nf_compliant = check_2nf(df_1nf, primary_keys, cleaned_fds_input)

    if not is_2nf_compliant:
        df_2nf = normalize_to_2nf(df_1nf, cleaned_fds_input, primary_keys)
        print("   ")
        print("2NF Normalized Tables")
        for i, table in enumerate(df_2nf):
            print(f"Table {i + 1}:")
            print(table.head(3))
            print("  ")
    else:
        print("The dataset is already in 2NF.")
        df_2nf = [df_1nf]
    print("   ")

    print("========== Is the table 3NF compliant? ==========")
    is_3nf_compliant = check_3nf(df_2nf, primary_keys, cleaned_fds_input)

    if not is_3nf_compliant:
        df_3nf = normalize_to_3nf(df_2nf, cleaned_fds_input, primary_keys)
        print("  ")
        print("3NF Normalized Tables")
        for i, table in enumerate(df_3nf):
            print(f"Table {i + 1}:")
            print(table)
            print("   ")
    else:
        print("The dataset is already in 3NF.")
        df_3nf = df_2nf
    print("   ")

    print("========== Is the table BCNF compliant? ==========")
    df_bcnf = normalize_to_bcnf(df_1nf, cleaned_fds_input)
    print(" ")

    print("========== All tables transferred to MySQL ==========")
    mydbase = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd='1234',)
    mycursor = mydbase.cursor()

    # Creating database
    mycursor.execute("CREATE DATABASE IF NOT EXISTS project_database")
    mycursor.execute("USE project_database")
    create_and_populate_tables(mycursor, df_bcnf)
    # Commit the changes to the database
    mydbase.commit()
    print(" ")

    # Close the connection
    mydbase.close()


if __name__ == "__main__":
    main()
