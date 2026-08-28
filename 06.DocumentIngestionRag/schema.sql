CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS document_chunks(
  id uuid PRIMARY KEY,
  document_id varchar(200) NOT NULL,
  title varchar(500) NOT NULL,
  section varchar(500),
  text text NOT NULL,
  country varchar(20),
  department varchar(100),
  version varchar(50),
  status varchar(30) NOT NULL DEFAULT 'Active',
  source_uri text,
  embedding vector(768) NOT NULL
);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);
