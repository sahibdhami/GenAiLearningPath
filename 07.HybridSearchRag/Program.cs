using GenAiLearning.HybridSearchRag;

var builder = WebApplication.CreateBuilder(args);

var options = builder.Configuration.GetSection("GoogleCloud").Get<GoogleCloudOptions>()!;
var client = GeminiClientFactory.Create(options);
var embeddings = new GeminiEmbeddingService(client, options);

await using var dataSource = PostgresFactory.Build(builder.Configuration.GetConnectionString("RagDatabase")!);

var store = new PostgresVectorStore(dataSource);
var retrieval = new HybridRetrievalService(embeddings, store);
var app = builder.Build();

app.MapGet("/search", async (string q) => Results.Ok(await retrieval.RetrieveAsync(q)));

app.MapGet("/ask", async (string q) =>
{
    var docs = await retrieval.RetrieveAsync(q, 5);

    var context = string.Join(
        """


        ---


        """,
        docs.Select(doc => $"""
            SOURCE_ID: {doc.Id}
            TITLE: {doc.Title}
            {doc.Text}
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
