namespace GenAiLearning.RerankingCitationsRag;

public static class ReciprocalRankFusion
{
    public static IReadOnlyList<RetrievalResult> Fuse(
        IReadOnlyList<RetrievalResult> vector,
        IReadOnlyList<RetrievalResult> lexical,
        int topK,
        int k = 60)
    {
        var scores = new Dictionary<Guid, double>();
        var documents = new Dictionary<Guid, RetrievalResult>();

        Accumulate(vector);
        Accumulate(lexical);

        return scores
            .OrderByDescending(entry => entry.Value)
            .Take(topK)
            .Select(entry => documents[entry.Key])
            .ToList();

        void Accumulate(IReadOnlyList<RetrievalResult> results)
        {
            for (var rank = 0; rank < results.Count; rank++)
            {
                var result = results[rank];

                scores[result.Id] = scores.GetValueOrDefault(result.Id) + 1d / (k + rank + 1);
                documents[result.Id] = result;
            }
        }
    }
}

public interface IReranker
{
    Task<IReadOnlyList<RetrievalResult>> RerankAsync(
        string query,
        IReadOnlyList<RetrievalResult> candidates,
        int topK,
        CancellationToken cancellationToken = default);
}

public sealed class LightweightLexicalReranker : IReranker
{
    public Task<IReadOnlyList<RetrievalResult>> RerankAsync(
        string query,
        IReadOnlyList<RetrievalResult> candidates,
        int topK,
        CancellationToken cancellationToken = default)
    {
        var terms = query
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        var ranked = candidates
            .Select(candidate => new
            {
                Candidate = candidate,
                Score = terms.Count(term =>
                    candidate.Text.Contains(term, StringComparison.OrdinalIgnoreCase) ||
                    candidate.Title.Contains(term, StringComparison.OrdinalIgnoreCase))
            })
            .OrderByDescending(scored => scored.Score)
            .Take(topK)
            .Select(scored => scored.Candidate)
            .ToList();

        return Task.FromResult<IReadOnlyList<RetrievalResult>>(ranked);
    }
}

public sealed class RetrievalPipeline(
    IEmbeddingService embeddings,
    PostgresVectorStore store,
    IReranker reranker)
{
    public async Task<IReadOnlyList<RetrievalResult>> RetrieveAsync(
        string query,
        CancellationToken cancellationToken = default)
    {
        var queryVector = await embeddings.GenerateQueryEmbeddingAsync(query, cancellationToken);

        var vectorTask = store.SearchVectorAsync(queryVector, 20, cancellationToken);
        var lexicalTask = store.SearchTextAsync(query, 20, cancellationToken);

        await Task.WhenAll(vectorTask, lexicalTask);

        var vectorResults = (await vectorTask).Select(hit => hit.Result).ToList();
        var fused = ReciprocalRankFusion.Fuse(vectorResults, await lexicalTask, 20);

        return await reranker.RerankAsync(query, fused, 5, cancellationToken);
    }
}
