using System.Text.Json;
using System.Text.Json.Nodes;
using GenAiLearning.RerankingCitationsRag;
using Google.GenAI.Types;

var builder = WebApplication.CreateBuilder(args);

var options = builder.Configuration.GetSection("GoogleCloud").Get<GoogleCloudOptions>()!;
var client = GeminiClientFactory.Create(options);
var embeddings = new GeminiEmbeddingService(client, options);

await using var dataSource = PostgresFactory.Build(builder.Configuration.GetConnectionString("RagDatabase")!);

var store = new PostgresVectorStore(dataSource);
var pipeline = new RetrievalPipeline(embeddings, store, new LightweightLexicalReranker());
var app = builder.Build();

app.MapGet("/ask", async (string q) =>
{
    var docs = await pipeline.RetrieveAsync(q);
    var allowedIds = docs.Select(doc => doc.Id.ToString()).ToHashSet();

    var context = string.Join(
        """


        ---


        """,
        docs.Select(doc => $"""
            SOURCE_ID: {doc.Id}
            TITLE: {doc.Title}
            SECTION: {doc.Section}
            {doc.Text}
            """));

    const string schema = """
        {
          "type": "object",
          "properties": {
            "answer": { "type": "string" },
            "sourceIds": { "type": "array", "items": { "type": "string" } }
          },
          "required": ["answer", "sourceIds"]
        }
        """;

    var config = new GenerateContentConfig
    {
        ResponseMimeType = "application/json",
        ResponseJsonSchema = JsonNode.Parse(schema),
        Temperature = 0.1
    };

    var prompt = $"""
        Answer only from supplied sources and cite SOURCE_ID values.

        CONTEXT:
        {context}

        QUESTION:
        {q}
        """;

    var response = await client.Models.GenerateContentAsync(options.GeminiModel, prompt, config);

    var parsed = JsonSerializer.Deserialize<CitedAnswer>(
        response.Candidates[0].Content.Parts[0].Text!,
        new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;

    var validIds = parsed.SourceIds.Where(allowedIds.Contains).ToArray();

    return Results.Ok(new
    {
        parsed.Answer,
        sourceIds = validIds,
        sources = docs
            .Where(doc => validIds.Contains(doc.Id.ToString()))
            .Select(doc => new { doc.Id, doc.Title, doc.Section, doc.SourceUri })
    });
});

app.Run();

public sealed record CitedAnswer(string Answer, string[] SourceIds);
