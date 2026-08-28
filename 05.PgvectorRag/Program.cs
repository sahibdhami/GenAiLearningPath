using GenAiLearning.PgvectorRag;

var builder = WebApplication.CreateBuilder(args);

var options = builder.Configuration.GetSection("GoogleCloud").Get<GoogleCloudOptions>()!;
var client = GeminiClientFactory.Create(options);
var embedding = new GeminiEmbeddingService(client, options);

await using var dataSource = PostgresFactory.Build(builder.Configuration.GetConnectionString("RagDatabase")!);

var store = new PostgresVectorStore(dataSource);

var app = builder.Build();

app.MapGet("/search", async (string q) =>
{
    var queryVector = await embedding.GenerateQueryEmbeddingAsync(q);

    return Results.Ok(await store.SearchVectorAsync(queryVector, 5));
});

app.MapGet("/ask", async (string q) =>
{
    var queryVector = await embedding.GenerateQueryEmbeddingAsync(q);
    var hits = await store.SearchVectorAsync(queryVector, 5);

    var context = string.Join(
        """


        ---


        """,
        hits.Select(hit => $"""
            SOURCE: {hit.Result.Title}
            {hit.Result.Text}
            """));

    var prompt = $"""
        Answer only from supplied context.

        CONTEXT:
        {context}

        QUESTION:
        {q}
        """;

    var response = await client.Models.GenerateContentAsync(options.GeminiModel, prompt);

    return Results.Ok(new
    {
        answer = response.Candidates[0].Content.Parts[0].Text,
        sources = hits.Select(hit => hit.Result.Title)
    });
});

app.Run();
