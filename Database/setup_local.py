import sqlite3

connection = sqlite3.connect('local.db')

cursor = connection.cursor()

schema = ""

with open('schema_local.sql', 'r') as schema_file:
    schema = schema_file.read()

cursor.executescript(schema)

connection.commit()
cursor.close()
connection.close()
