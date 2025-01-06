from flask import Flask, request, jsonify, render_template, g
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import mysql.connector
import psycopg2
import sqlite3
import json
import os

app = Flask(__name__)

# Global variables for vectorstore
vectorstore = None

# Function to connect to the database
def connect_database(db_type, username, port, host, password, database):
    try:
        if db_type == "MySQL":
            conn = mysql.connector.connect(
                host=host,
                user=username,
                password=password,
                database=database,
                port=int(port)
            )
        elif db_type == "PostgreSQL":
            conn = psycopg2.connect(
                host=host,
                user=username,
                password=password,
                dbname=database,
                port=int(port)
            )
        elif db_type == "SQLite":
            conn = sqlite3.connect(database)
        else:
            return None, "Unsupported database type"

        return conn, "Connection successful"
    except Exception as err:
        return None, f"Failed to connect to database: {err}"

# Get database connection from Flask's g object
def get_db():
    if 'db' not in g:
        return None
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Function to execute queries
def run_query(query):
    conn = get_db()
    if not conn:
        return "Please connect to the database"

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Exception as err:
        return f"Error executing query: {err}"

# Function to get the database schema
def get_database_schema():
    conn = get_db()
    if not conn:
        return "Please connect to the database"

    try:
        cursor = conn.cursor()
        db_type = g.db_type
        if db_type == "MySQL":
            cursor.execute("SHOW TABLES;")
        elif db_type == "PostgreSQL":
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        elif db_type == "SQLite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        else:
            return "Unsupported database type"

        tables = cursor.fetchall()
        schema = {}
        for (table,) in tables:
            if db_type == "MySQL" or db_type == "PostgreSQL":
                cursor.execute(f"DESCRIBE {table};")
            elif db_type == "SQLite":
                cursor.execute(f"PRAGMA table_info({table});")
            schema[table] = cursor.fetchall()

        return schema
    except Exception as err:
        return f"Error retrieving schema: {err}"

# Initialize Vectorstore for RAG
def initialize_vectorstore(data_dir):
    global vectorstore
    if not os.path.exists(data_dir):
        return "Data directory does not exist"

    documents = []
    for file_name in os.listdir(data_dir):
        with open(os.path.join(data_dir, file_name), 'r') as f:
            content = f.read()
            documents.append(content)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(documents, embeddings)
    return "Vectorstore initialized"

# Define LLM and templates
llm = ChatOllama(model="llama3")

# Fine-tuning placeholder function
def fine_tune_llm(training_data):
    try:
        # This is a placeholder for the fine-tuning process
        # In practice, you would use tools like Hugging Face's Transformers or Ollama APIs
        fine_tuned_model_path = "fine_tuned_llm_model_path"
        return f"Fine-tuning successful. Model saved at {fine_tuned_model_path}"
    except Exception as err:
        return f"Fine-tuning failed: {err}"

# Get query from LLM or Vectorstore (RAG)
def get_query_from_llm(question):
    schema = get_database_schema()

    if isinstance(schema, str):
        return schema

    if vectorstore:
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA(llm=llm, retriever=retriever)
        return qa_chain.run(question)

    schema_text = "\n".join(
        f"Table: {table}\n" + "\n".join(f"  {column}" for column in columns)
        for table, columns in schema.items()
    )

    template = """below is the schema of {db_type} database, read the schema carefully about the table and column names. Also take care of table or column name case sensitivity.
    Finally answer user's question in the form of SQL query.

    {schema}

    please only provide the SQL query and nothing else

    your turn :
    question: {question}
    SQL query :
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    response = chain.invoke({
        "question": question,
        "schema": schema_text,
        "db_type": g.db_type
    })
    return response.content

@app.route('/fine_tune', methods=['POST'])
def fine_tune():
    data = request.json
    training_data = data.get("training_data")
    if not training_data:
        return jsonify({"error": "Training data not provided."})

    message = fine_tune_llm(training_data)
    return jsonify({"message": message})

@app.route('/initialize_rag', methods=['POST'])
def initialize_rag():
    data = request.json
    data_dir = data.get("data_dir")
    if not data_dir:
        return jsonify({"error": "Data directory not provided."})

    message = initialize_vectorstore(data_dir)
    return jsonify({"message": message})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/connect', methods=['POST'])
def connect():
    data = request.json
    conn, message = connect_database(
        db_type=data['db_type'],
        username=data.get('username', ''),
        port=data.get('port', ''),
        host=data.get('host', ''),
        password=data.get('password', ''),
        database=data.get('database', '')
    )

    if conn:
        g.db = conn
        g.db_type = data['db_type']

    return jsonify({"message": message})

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data['question']

    if 'db' not in g:
        return jsonify({"error": "Please connect to the database first."})

    query = get_query_from_llm(question)
    result = run_query(query)
    return jsonify({"query": query, "result": result})

if __name__ == '__main__':
    app.run(debug=True)
