# Databricks notebook source
pip install pypdf transformers databricks-vectorsearch langchain databricks-sdk --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#Fill this out with the catalog & schema where the data is stored and where all objects will be created (eg: tables, models etc)

catalog = "catalog_name"
schema = "schema_name"
spark.sql(f"""USE CATALOG {catalog}""")
spark.sql(f"""USE DATABASE {schema}""")

#Replace the names below with the names of the endpoints for embedding model & chat model that would have been pre provisioned within the workspace. Replace the name for vector search endpoint & the name of the table to write to. Create a volume in the selected catalog & schema and copy the path into volume name below
embedding_endpoint_name = "bge_m3"
chat_endpoint_name = "llama_3_instruct"
vector_search_endpoint_name = "shared-endpoint"
table_name = "table_embeddings"
volume_path = "/Volumes/catalog_name/schema_name/volume_name/"

# COMMAND ----------

#Create functions to parse in pdf files and return a dictionary with information on the file, number of pages, and the text of each page.

import pandas as pd
from pypdf import PdfReader

def get_file_path (path):
 output = dbutils.fs.ls(path)
 file_path = []
 for i in output:
  if i.path.startswith("dbfs:"):
    file_path.append(i.path[5:])
 return file_path

def get_page_number(file):
    reader = PdfReader(file)
    return len(reader.pages)

def get_page_content(file):
  reader = PdfReader(file)
  text_list = []
  for page in reader.pages:
    text_list.append(page.extract_text())
  return text_list

    
def get_content_dict(path):
  content = []
  files = get_file_path(path)

  for file in files:
    content_dict = {}
    content_dict["file_path"] = file
    content_dict["number_of_pages"] = get_page_number(file)
    content_dict["content"] = get_page_content(file)
    content.append(content_dict)
  
  return content


# COMMAND ----------

#Ingest files from the volume specified above and create a dataframe from the file content

pdf_dict = get_content_dict(volume_path)
pdf_df = spark.createDataFrame(pdf_dict)

# COMMAND ----------


#Chunk the text extracted from files into smaller chunks to fit into the context length for the LLM.

from pyspark.sql.functions import pandas_udf
from transformers import  AutoTokenizer
from langchain.text_splitter  import RecursiveCharacterTextSplitter 
from pyspark.sql import types as T


tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=500, chunk_overlap=50)

def get_chunks(result):
  chunks = []
  previous_chunk = ""

  for c in result:
    #Check if the chunks are small, if so add to them until they reach the midpoint of the max token size
    if len(tokenizer.encode(previous_chunk + c)) <= 500/2:
      previous_chunk += c + "\n"

    #Once they reach the midpoint, if the chunk itself is large, then just use the splitter to split it and add to the chunk list. Else, if the previous chunk is large, use the splitter to split them out and add to list, and reset the previous chunk to the current chunk.

    elif len(tokenizer.encode(previous_chunk + c)) > 500/2:
      if len(tokenizer.encode(previous_chunk)) == 1 or c == result[-1]:
        chunks.extend(text_splitter.split_text(c.strip()))

      else:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
        previous_chunk = c
    
    elif c == result[-1]:
      chunks.extend(text_splitter.split_text(c.strip()))

  return chunks

@pandas_udf(T.ArrayType(T.StringType()))
def get_chunks_udf(result: pd.Series) -> pd.Series:
  return result.apply(get_chunks)

# COMMAND ----------

pdf_df.withColumn("chunks", get_chunks_udf("content")).display()

# COMMAND ----------

from pyspark.sql import functions as F
pdf_chunks = pdf_df.withColumn("chunks", get_chunks_udf("content")) \
                   .withColumn("chunk_exploded",F.explode("chunks")) \
                   .dropna(how = "any", subset = ['chunk_exploded']) \
                   .drop("chunks","content")

# COMMAND ----------

pdf_chunks.display()

# COMMAND ----------

#Convert chunked text into embeddings

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

def compute_embeddings(batch):
  response = deploy_client.predict(endpoint=embedding_endpoint_name, inputs={"input": batch})
  return response['data'][0]['embedding']

# COMMAND ----------

def apply_embedding(pdf_chunks_list):
  embedding_list = []
  for row in pdf_chunks_list:
    embedding_dict = {}
    embedding_dict['file_path'] = row['file_path']
    embedding_dict['number_of_pages'] = row['number_of_pages']
    embedding_dict['chunk_exploded'] = row['chunk_exploded']
    embedding_dict['embedding'] = compute_embeddings(row['chunk_exploded'])
    embedding_list.append(embedding_dict)
  
  return embedding_list

# COMMAND ----------

pdf_chunks_to_embed_list = [row.asDict() for row in pdf_chunks.collect()]

# COMMAND ----------

embeddings = spark.createDataFrame(apply_embedding(pdf_chunks_to_embed_list)).withColumn("id", F.monotonically_increasing_id())
embeddings.display()

# COMMAND ----------

embeddings.write.option("mergeSchema", "true").mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

spark.sql(f"""ALTER TABLE {catalog}.{schema}.{table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

# COMMAND ----------

#Create a vector search index

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
vsc = VectorSearchClient()

vs_index_fullname = f"{catalog}.{schema}.{table_name}_index"

def create_or_update_index():
  if vs_index_fullname not in [i['name'] for i in vsc.list_indexes(vector_search_endpoint_name)['vector_indexes']]:
    vsc.create_delta_sync_index(
      endpoint_name=vector_search_endpoint_name,
      index_name=vs_index_fullname,
      source_table_name=f"{catalog}.{schema}.{table_name}",
      primary_key="id",
      pipeline_type="TRIGGERED",
      embedding_source_column = "chunk_exploded",
      embedding_model_endpoint_name = embedding_endpoint_name
    )
    print("Index created")

  else:
    vsc.get_index(vector_search_endpoint_name, vs_index_fullname).sync()
    print("Index updated")

create_or_update_index()

# COMMAND ----------

#Query the vector index to return information ingested from the pdf files

import mlflow.deployments

query = "What is the energy crisis for data centres and what would help resolve it?"
number_of_rows_searched = 5

def query_endpoint(query, number_of_rows):
  results = vsc.get_index(vector_search_endpoint_name, vs_index_fullname).similarity_search(
    query_text = query,
    columns = ['id','chunk_exploded'],
    num_results=number_of_rows
    )
  
  results_array = results['result']['data_array']
  result_list = []
  
  for result in results_array:
    result_list.append(result[1])
  
  deploy_client = mlflow.deployments.get_deploy_client("databricks")
  response = deploy_client.predict(endpoint=chat_endpoint_name, inputs={"messages": [{"role":"user","content":f"Summarise the list of issues based on {query} in 200 words or less: '{result_list}"}]})
  return response['choices'][0]['message']['content']

print(query_endpoint(query,number_of_rows_searched))


# COMMAND ----------

#Optional exercises:
#Use langchain to connect these different steps in one API call
#Log lanchain and save as a model on Unity Catalog
#Deploy the model as a serving endpoint
#Create a UI to query the model
