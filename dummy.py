# Databricks notebook source
# MAGIC %md
# MAGIC # Codebase Copilot -- Chunk Ingestion into Delta
# MAGIC
# MAGIC Loads the JSON exported by `export_chunks_for_databricks.py` and writes
# MAGIC it into a managed Delta table under Unity Catalog. Run cells top to
# MAGIC bottom. Cell 2 discovers your workspace's actual catalog/schema names
# MAGIC rather than assuming one -- Free Edition workspaces can differ.

# COMMAND ----------

# MAGIC %md ## 1. Upload the JSON first
# MAGIC Before running this notebook: upload `chunks_export.json` (produced
# MAGIC locally by `export_chunks_for_databricks.py`) into a Unity Catalog
# MAGIC Volume, via **Catalog Explorer -> your catalog -> your schema ->
# MAGIC Volumes -> Upload**. Note the resulting path, e.g.:
# MAGIC `/Volumes/<catalog>/<schema>/<volume>/chunks_export.json`
# MAGIC
# MAGIC Set that path (and your target catalog/schema/table names) below.

# COMMAND ----------

dbutils.widgets.text("volume_json_path", "/Volumes/workspace/default/uploads/chunks_export.json")
dbutils.widgets.text("catalog_name", "workspace")
dbutils.widgets.text("schema_name", "default")
dbutils.widgets.text("table_name", "codebase_chunks")

volume_json_path = dbutils.widgets.get("volume_json_path")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
table_name = dbutils.widgets.get("table_name")

# COMMAND ----------

# MAGIC %md ## 2. Discover what actually exists in your workspace
# MAGIC Run this BEFORE trusting the widget defaults above -- confirms your
# MAGIC real catalog/schema names instead of guessing.

# COMMAND ----------

display(spark.sql("SHOW CATALOGS"))

# COMMAND ----------

# Once you know your real catalog name, list its schemas:
display(spark.sql(f"SHOW SCHEMAS IN {catalog_name}"))

# COMMAND ----------

# MAGIC %md ## 3. Load the JSON and inspect it before writing anything

# COMMAND ----------

df = spark.read.option("multiline", "true").json(volume_json_path)
display(df)
print(f"Rows: {df.count()}")
df.printSchema()

# COMMAND ----------

# MAGIC %md ## 4. Write as a managed Delta table
# MAGIC `saveAsTable` with no format specified defaults to Delta -- this is
# MAGIC the actual "integrate with Databricks" step. `mode("overwrite")` lets
# MAGIC you re-run this notebook safely if you re-export chunks later (e.g.
# MAGIC after adding files to sample_repo) without manually dropping the table.

# COMMAND ----------

full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

(df.write
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable(full_table_name))

print(f"Wrote {df.count()} rows to {full_table_name}")

# COMMAND ----------

# MAGIC %md ## 5. Confirm it's queryable with plain SQL
# MAGIC This is the payoff -- once it's a Delta table, it's queryable by
# MAGIC anyone/anything that speaks SQL, including a Genie space pointed at
# MAGIC this table, without touching Python or your original repo at all.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT file, name, type, code_length, has_docstring
# MAGIC FROM IDENTIFIER(:full_table_name)
# MAGIC ORDER BY code_length DESC

# COMMAND ----------

# MAGIC %md ## 6. Example analytical queries -- good live-demo material
# MAGIC These are the kind of question a Genie space could answer in plain
# MAGIC English once pointed at this table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Functions with no docstring at all
# MAGIC SELECT file, name FROM IDENTIFIER(:full_table_name)
# MAGIC WHERE has_docstring = false AND type = 'function'

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Which file has the most functions?
# MAGIC SELECT file, COUNT(*) as function_count
# MAGIC FROM IDENTIFIER(:full_table_name)
# MAGIC WHERE type = 'function'
# MAGIC GROUP BY file
# MAGIC ORDER BY function_count DESC

# COMMAND ----------

# MAGIC %md ## 7. Create a Vector Search index on top of this Delta table
# MAGIC This is the actual FAISS replacement. Databricks Vector Search creates
# MAGIC an auto-updating index synced FROM the Delta table -- so instead of
# MAGIC rebuilding a local FAISS index every time you re-index a repo, you
# MAGIC write to the Delta table (Section 4 above) and the index updates
# MAGIC itself. This requires Change Data Feed enabled on the source table.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch databricks-langchain
dbutils.library.restartPython()

# COMMAND ----------

# Re-declare widgets after the Python restart above (restartPython clears state)
dbutils.widgets.text("catalog_name", "workspace")
dbutils.widgets.text("schema_name", "default")
dbutils.widgets.text("table_name", "codebase_chunks")
dbutils.widgets.text("endpoint_name", "codebase_copilot_vs_endpoint")
dbutils.widgets.text("index_name", "codebase_chunks_index")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
table_name = dbutils.widgets.get("table_name")
endpoint_name = dbutils.widgets.get("endpoint_name")
index_name = dbutils.widgets.get("index_name")

full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
full_index_name = f"{catalog_name}.{schema_name}.{index_name}"

# COMMAND ----------

# MAGIC %md ### 7a. Enable Change Data Feed on the source table
# MAGIC Required for a Delta Sync Index on a standard endpoint -- this is
# MAGIC what lets Vector Search detect row-level changes incrementally
# MAGIC instead of rescanning the whole table on every sync.

# COMMAND ----------

spark.sql(f"""
ALTER TABLE {full_table_name}
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print(f"Change Data Feed enabled on {full_table_name}")

# COMMAND ----------

# MAGIC %md ### 7b. Add a text column for embedding, and a stable primary key
# MAGIC Vector Search needs one column to embed and a primary key column.
# MAGIC `id` (file::function_name) is already unique -- reuse it. For the
# MAGIC embedding text, reuse the same docstring+code concatenation idea
# MAGIC from `_chunks_to_documents` in retriever.py, so retrieval quality
# MAGIC here matches what you already validated locally.

# COMMAND ----------

from pyspark.sql.functions import concat_ws, coalesce, lit

spark.sql(f"""
ALTER TABLE {full_table_name} ADD COLUMN IF NOT EXISTS text_for_embedding STRING
""")

spark.sql(f"""
UPDATE {full_table_name}
SET text_for_embedding = concat_ws('\\n',
    concat('# ', file, '::', name, ' (', type, ')'),
    coalesce(docstring, ''),
    code
)
""")

display(spark.sql(f"SELECT id, text_for_embedding FROM {full_table_name} LIMIT 3"))

# COMMAND ----------

# MAGIC %md ### 7c. Create the Vector Search endpoint
# MAGIC One-time setup -- if it already exists from a previous run, this
# MAGIC cell will error harmlessly; check the endpoint list in the
# MAGIC Databricks UI (Compute -> Vector Search) if unsure.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient()

try:
    vs_client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
    print(f"Created endpoint: {endpoint_name}")
except Exception as e:
    print(f"Endpoint may already exist, or creation failed: {e}")

# COMMAND ----------

# MAGIC %md ### 7d. Create the Delta Sync Index
# MAGIC Using Databricks-managed embeddings (`embedding_source_column` +
# MAGIC `embedding_model_endpoint_name`) means Databricks computes and
# MAGIC maintains the embeddings for you -- no separate embedding step to
# MAGIC maintain, unlike the local FAISS path where retriever.py calls
# MAGIC HuggingFaceEmbeddings explicitly.

# COMMAND ----------

index = vs_client.create_delta_sync_index_and_wait(
    endpoint_name=endpoint_name,
    source_table_name=full_table_name,
    index_name=full_index_name,
    pipeline_type="TRIGGERED",   # sync on demand, not continuous -- cheaper for a demo/sprint
    primary_key="id",
    embedding_source_column="text_for_embedding",
    embedding_model_endpoint_name="databricks-gte-large-en",  # Databricks-hosted embedding model
)
print(f"Index created: {full_index_name}")

# COMMAND ----------

# MAGIC %md ### 7e. Test it with a real similarity search query
# MAGIC This is the direct equivalent of `retriever.search(query, k=4)` in
# MAGIC your local FAISS code -- same idea, running on Databricks instead.

# COMMAND ----------

results = index.similarity_search(
    query_text="where is division handled",
    columns=["id", "file", "name", "type", "code"],
    num_results=4,
)
import json
print(json.dumps(results, indent=2))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Longest functions -- candidates for refactoring, and a callback
# MAGIC -- to your earlier chunking-strategy comparison work
# MAGIC SELECT file, name, code_length
# MAGIC FROM IDENTIFIER(:full_table_name)
# MAGIC ORDER BY code_length DESC
# MAGIC LIMIT 5