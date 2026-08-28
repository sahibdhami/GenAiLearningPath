using GenAiLearning.AddedVectorToRag;

var builder = WebApplication.CreateBuilder(args);

var options = builder.Configuration.GetSection("GoogleCloud").Get<GoogleCloudOptions>()!;
var client = GeminiClientFactory.Create(options);
var embedding = new GeminiEmbeddingService(client, options);

IVectorStore store = new InMemoryVectorStore();

var seedChunks = new[]
{
    ("c1", "policy-17", "Lost Package Policy", "Packages with no tracking movement for 48 hours must be escalated."),
    ("c2", "policy-25", "Password Reset", "Employees reset passwords in the identity portal."),
    ("c3", "policy-31", "Damaged Shipment", "Damaged shipments require photos and a claims submission.")
};

foreach (var (id, documentId, title, text) in seedChunks)
{
    var vector = await embedding.GenerateDocumentEmbeddingAsync(text);

    store.Add(new DocumentChunk(id, documentId, title, text, vector));
}

var search = new SemanticSearchService(embedding, store);
var app = builder.Build();

app.MapGet("/search", async (string q) => Results.Ok(await search.SearchAsync(q)));

app.MapGet("/ask", async (string q) =>
{
    var hits = await search.SearchAsync(q);

    var context = string.Join(
        """


        ---


        """,
        hits.Select(hit => $"""
            SOURCE: {hit.Title}
            {hit.Text}
            """));

    var prompt = $"""
        Answer only from context.

        CONTEXT:
        {context}

        QUESTION:
        {q}
        """;

    var response = await client.Models.GenerateContentAsync(options.GeminiModel, prompt);

    return Results.Ok(new { answer = response.Candidates[0].Content.Parts[0].Text });
});

app.Run();
