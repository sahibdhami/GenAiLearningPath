using Google.GenAI;
using Google.GenAI.Types;

namespace GenAiLearning.PgvectorRag;

public interface IEmbeddingService
{
    Task<float[]> GenerateDocumentEmbeddingAsync(string text, CancellationToken ct = default);
    Task<float[]> GenerateQueryEmbeddingAsync(string text, CancellationToken ct = default);
}

public sealed class GeminiEmbeddingService : IEmbeddingService
{
    private readonly Client _client;
    private readonly GoogleCloudOptions _options;

    public GeminiEmbeddingService(Client client, GoogleCloudOptions options)
    {
        _client = client;
        _options = options;
    }

    public Task<float[]> GenerateDocumentEmbeddingAsync(string text, CancellationToken ct = default) =>
        GenerateAsync(text, "RETRIEVAL_DOCUMENT", ct);

    public Task<float[]> GenerateQueryEmbeddingAsync(string text, CancellationToken ct = default) =>
        GenerateAsync(text, "RETRIEVAL_QUERY", ct);

    private async Task<float[]> GenerateAsync(string text, string taskType, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var response = await _client.Models.EmbedContentAsync(
            model: _options.EmbeddingModel,
            contents: text,
            config: new EmbedContentConfig
            {
                TaskType = taskType,
                OutputDimensionality = _options.EmbeddingDimensions,
                AutoTruncate = true
            });

        var values = response.Embeddings?.FirstOrDefault()?.Values
            ?? throw new InvalidOperationException("Embedding model returned no vector.");
        return values.Select(v => (float)v).ToArray();
    }
}
