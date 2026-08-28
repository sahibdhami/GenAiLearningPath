using Google.GenAI;
using GenAiLearning.IntroducingRag;

var builder = WebApplication.CreateBuilder(args);
var options = builder.Configuration.GetSection("GoogleCloud").Get<GoogleCloudOptions>()!;
var client = GeminiClientFactory.Create(options);
var embeddings = new GeminiEmbeddingService(client, options);
var chunks = new List<DocumentChunk>();

async Task SeedAsync()
{
    var documents = new[]
    {
        ("policy-17", "Lost Package Policy", "Packages with no tracking movement for 48 hours must be escalated to the Package Investigation Team."),
        ("policy-25", "Password Reset Procedure", "Employees reset forgotten passwords through the corporate identity portal."),
        ("policy-31", "Damaged Shipment Procedure", "Damaged shipments must be photographed and submitted to the claims processing team.")
    };

    foreach (var (id, title, text) in documents)
    {
        var vector = await embeddings.GenerateDocumentEmbeddingAsync(text);

        chunks.Add(new DocumentChunk(id, title, text, vector));
    }
}

await SeedAsync();

var app = builder.Build();

app.MapGet("/search", async (string q) =>
{
    var queryVector = await embeddings.GenerateQueryEmbeddingAsync(q);

    var hits = chunks
        .Select(chunk => new SearchHit(chunk, VectorMath.CosineSimilarity(queryVector, chunk.Embedding)))
        .OrderByDescending(hit => hit.Similarity)
        .Take(3);

    return Results.Ok(hits.Select(hit => new
    {
        hit.Chunk.Id,
        hit.Chunk.Title,
        hit.Chunk.Text,
        hit.Similarity
    }));
});

app.MapGet("/ask", async (string q) =>
{
    var queryVector = await embeddings.GenerateQueryEmbeddingAsync(q);

    var context = chunks
        .Select(chunk => new SearchHit(chunk, VectorMath.CosineSimilarity(queryVector, chunk.Embedding)))
        .OrderByDescending(hit => hit.Similarity)
        .Take(3);

    var joined = string.Join(
        """


        ---


        """,
        context.Select(hit => $"""
            SOURCE: {hit.Chunk.Title}
            {hit.Chunk.Text}
            """));

    var prompt = $"""
        Use only the context. If insufficient, say so.

        CONTEXT:
        {joined}

        QUESTION:
        {q}
        """;

    var response = await client.Models.GenerateContentAsync(options.GeminiModel, prompt);

    return Results.Ok(new { answer = response.Candidates[0].Content.Parts[0].Text });
});

app.Run();
