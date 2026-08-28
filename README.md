# GenAI Learning Path — .NET / Gemini / RAG

This solution captures the progressive application we built in the conversation. Each project is self-contained so you can compare changes directly.

| # | Project | What is introduced |
|---|---|---|
| 01 | DirectGemini | First direct Gemini call from ASP.NET Core |
| 02 | StructuredGemini | System instruction, temperature, JSON schema, typed output |
| 03 | IntroducingRag | Embeddings, cosine similarity, in-memory semantic search, first RAG |
| 04 | AddedVectorToRag | `IVectorStore` abstraction and separate semantic-search layer |
| 05 | PgvectorRag | PostgreSQL + pgvector, cosine distance, HNSW |
| 06 | DocumentIngestionRag | Ingestion, chunking, metadata, deterministic chunk identity |
| 07 | HybridSearchRag | PostgreSQL full-text search + vector search + RRF |
| 08 | RerankingCitationsRag | Reranker abstraction + validated source citations |
| 09 | ToolCallingFoundation | Gemini function/tool calling and operational tools |

## Prerequisites

- .NET 8 SDK or newer
- Google Cloud project with Gemini/Vertex AI access
- Google Cloud ADC for local development: `gcloud auth application-default login`
- For projects 05–08: Docker (or PostgreSQL with pgvector installed)

Set `GoogleCloud:ProjectId` in each project's `appsettings.json`. The sample uses `global`, `gemini-2.5-flash`, `gemini-embedding-001`, and 768-dimensional embeddings. Verify model availability for your GCP project/location before running.

## Build

```bash
dotnet restore GenAiLearningPath.sln
dotnet build GenAiLearningPath.sln
```

## PostgreSQL / pgvector

```bash
docker compose up -d
```

For the milestone you are running, execute that project's `schema.sql` against `ragdb`, for example:

```bash
psql -h localhost -U postgres -d ragdb -f 05.PgvectorRag/schema.sql
```

Projects 06–08 use the same `document_chunks` concept; project 07/08 adds the generated full-text search column/index.

## Important learning note

The projects intentionally duplicate code instead of sharing a common library. This is deliberate: the goal is to make the architectural additions visible from one project to the next. In a production repository you would normally extract shared infrastructure, domain models, and provider integrations into reusable projects.

## Production hardening not yet implemented

Authentication/authorization, metadata-based access control, retries, rate limits, OpenTelemetry, model/RAG evaluation, prompt-injection defenses, human approval for write actions, idempotency, and bounded agent loops are intentionally left for the next lessons.
