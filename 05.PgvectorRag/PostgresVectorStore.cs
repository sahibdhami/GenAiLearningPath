using Npgsql;
using Pgvector;
using Pgvector.Npgsql;

namespace GenAiLearning.PgvectorRag;

public sealed record RetrievalResult(
    Guid Id,
    string DocumentId,
    string Title,
    string Section,
    string Text,
    string SourceUri);

public sealed record VectorSearchHit(RetrievalResult Result, double Similarity);

public sealed class PostgresVectorStore(NpgsqlDataSource dataSource)
{
    private readonly NpgsqlDataSource _dataSource = dataSource;

    public async Task AddAsync(
        Guid id,
        string documentId,
        string title,
        string section,
        string text,
        string country,
        string department,
        string version,
        string sourceUri,
        float[] embedding,
        CancellationToken cancellationToken = default)
    {
        const string sql = """
            INSERT INTO document_chunks
                (id, document_id, title, section, text, country, department, version, status, source_uri, embedding)
            VALUES
                (@id, @documentId, @title, @section, @text, @country, @department, @version, 'Active', @sourceUri, @embedding)
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                version = EXCLUDED.version,
                status = 'Active';
            """;

        await using var command = _dataSource.CreateCommand(sql);

        command.Parameters.AddWithValue("id", id);
        command.Parameters.AddWithValue("documentId", documentId);
        command.Parameters.AddWithValue("title", title);
        command.Parameters.AddWithValue("section", section);
        command.Parameters.AddWithValue("text", text);
        command.Parameters.AddWithValue("country", country);
        command.Parameters.AddWithValue("department", department);
        command.Parameters.AddWithValue("version", version);
        command.Parameters.AddWithValue("sourceUri", sourceUri);
        command.Parameters.AddWithValue("embedding", new Vector(embedding));

        await command.ExecuteNonQueryAsync(cancellationToken);
    }

    public async Task<IReadOnlyList<VectorSearchHit>> SearchVectorAsync(
        float[] embedding,
        int topK,
        CancellationToken cancellationToken = default)
    {
        const string sql = """
            SELECT id, document_id, title, section, text, source_uri,
                   1 - (embedding <=> @embedding) AS similarity
            FROM document_chunks
            WHERE status = 'Active'
            ORDER BY embedding <=> @embedding
            LIMIT @topK;
            """;

        await using var command = _dataSource.CreateCommand(sql);

        command.Parameters.AddWithValue("embedding", new Vector(embedding));
        command.Parameters.AddWithValue("topK", topK);

        var hits = new List<VectorSearchHit>();

        await using var reader = await command.ExecuteReaderAsync(cancellationToken);

        while (await reader.ReadAsync(cancellationToken))
        {
            var result = new RetrievalResult(
                reader.GetGuid(0),
                reader.GetString(1),
                reader.GetString(2),
                reader.IsDBNull(3) ? string.Empty : reader.GetString(3),
                reader.GetString(4),
                reader.IsDBNull(5) ? string.Empty : reader.GetString(5));

            hits.Add(new VectorSearchHit(result, reader.GetDouble(6)));
        }

        return hits;
    }
}

public static class PostgresFactory
{
    public static NpgsqlDataSource Build(string connectionString)
    {
        var builder = new NpgsqlDataSourceBuilder(connectionString);

        builder.UseVector();

        return builder.Build();
    }
}
