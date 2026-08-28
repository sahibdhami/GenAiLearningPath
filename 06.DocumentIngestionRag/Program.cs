using GenAiLearning.DocumentIngestionRag;
using System.Reflection.Metadata;

var builder = WebApplication.CreateBuilder(args);

var options = builder.Configuration.GetSection("GoogleCloud").Get<GoogleCloudOptions>()!;
var client = GeminiClientFactory.Create(options);
var embeddings = new GeminiEmbeddingService(client, options);

await using var dataSource = PostgresFactory.Build(builder.Configuration.GetConnectionString("RagDatabase")!);

var store = new PostgresVectorStore(dataSource);
var ingestion = new DocumentIngestionService(new ParagraphDocumentChunker(), embeddings, store);
var app = builder.Build();
//// Code to loop through all the files in a folder and sub folder.

//var folderPath = Path.Combine(AppContext.BaseDirectory, "documents");
//foreach (var filePath in Directory.EnumerateFiles(folderPath, "*.md", SearchOption.AllDirectories))
//{
//    var content = await File.ReadAllTextAsync(filePath);
//    await ingestion.IndexAsync(new SourceDocument(Guid.NewGuid().ToString(),filePath, content,"US", "tech", "v1", filePath));
//}

app.MapPost("/documents", async (SourceDocument document) =>
    Results.Ok(new { chunks = await ingestion.IndexAsync(document) }));

app.MapGet("/search", async (string q) =>
{
    var queryVector = await embeddings.GenerateQueryEmbeddingAsync(q);

    return Results.Ok(await store.SearchVectorAsync(queryVector, 5));
});

app.Run();
