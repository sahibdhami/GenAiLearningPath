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
  embedding vector(768) NOT NULL,
  search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(section,'') || ' ' || coalesce(text,''))) STORED
);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS document_chunks_search_idx ON document_chunks USING gin(search_vector);
