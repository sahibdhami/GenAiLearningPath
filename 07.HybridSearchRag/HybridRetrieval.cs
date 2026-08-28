namespace GenAiLearning.HybridSearchRag;

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

                scores[result.Id] = scores.GetValueOrDefault(result.Id) + 1.0 / (k + rank + 1);
                documents[result.Id] = result;
            }
        }
    }
}

public sealed class HybridRetrievalService(IEmbeddingService embeddings, PostgresVectorStore store)
{
    public async Task<IReadOnlyList<RetrievalResult>> RetrieveAsync(
        string query,
        int topK = 10,
        CancellationToken cancellationToken = default)
    {
        var queryVector = await embeddings.GenerateQueryEmbeddingAsync(query, cancellationToken);

        var vectorTask = store.SearchVectorAsync(queryVector, 20, cancellationToken);
        var lexicalTask = store.SearchTextAsync(query, 20, cancellationToken);

        await Task.WhenAll(vectorTask, lexicalTask);

        var vectorResults = (await vectorTask).Select(hit => hit.Result).ToList();

        return ReciprocalRankFusion.Fuse(vectorResults, await lexicalTask, topK);
    }
}
