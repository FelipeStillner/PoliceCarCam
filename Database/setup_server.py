import psycopg2

connection = psycopg2.connect(database="postgres", user="postgres", password="postgres", host="localhost", port=5432)

cursor = connection.cursor()

cursor.execute(open("schema_server.sql", "r").read())

cursor.execute(open("data.sql", "r").read())

cursor.close()

connection.commit()
