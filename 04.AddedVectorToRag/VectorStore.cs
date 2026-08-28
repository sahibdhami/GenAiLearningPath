namespace GenAiLearning.AddedVectorToRag;

public sealed record DocumentChunk(string Id, string DocumentId, string Title, string Text, float[] Embedding);

public sealed record SearchHit(string Id, string DocumentId, string Title, string Text, double Similarity);

public interface IVectorStore
{
    void Add(DocumentChunk chunk);

    IReadOnlyCollection<DocumentChunk> GetAll();
}

public sealed class InMemoryVectorStore : IVectorStore
{
    private readonly List<DocumentChunk> _items = [];

    public void Add(DocumentChunk chunk) => _items.Add(chunk);

    public IReadOnlyCollection<DocumentChunk> GetAll() => _items;
}

public static class VectorMath
{
    public static double CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0;
        double magnitudeA = 0;
        double magnitudeB = 0;

        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }

        return magnitudeA == 0 || magnitudeB == 0
            ? 0
            : dot / (Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB));
    }
}

public sealed class SemanticSearchService(IEmbeddingService embeddings, IVectorStore store)
{
    public async Task<IReadOnlyList<SearchHit>> SearchAsync(string query, int topK = 3)
    {
        var queryVector = await embeddings.GenerateQueryEmbeddingAsync(query);

        return store.GetAll()
            .Select(chunk => new SearchHit(
                chunk.Id,
                chunk.DocumentId,
                chunk.Title,
                chunk.Text,
                VectorMath.CosineSimilarity(queryVector, chunk.Embedding)))
            .OrderByDescending(hit => hit.Similarity)
            .Take(topK)
            .ToList();
    }
}
