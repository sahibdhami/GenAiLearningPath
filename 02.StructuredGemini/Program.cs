using System.Text.Json;
using System.Text.Json.Nodes;
using Google.GenAI;
using Google.GenAI.Types;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

var projectId = builder.Configuration["GoogleCloud:ProjectId"] ?? throw new InvalidOperationException("Set GoogleCloud:ProjectId.");
var location = builder.Configuration["GoogleCloud:Location"] ?? "global";
var model = builder.Configuration["GoogleCloud:GeminiModel"] ?? "gemini-2.5-flash";
var client = new Client(project: projectId, location: location, enterprise: true);

app.MapPost("/api/analyze", async (AnalyzeRequest request) =>
{
    const string schema = """
    {"type":"object","properties":{"category":{"type":"string"},"priority":{"type":"string"},"summary":{"type":"string"}},"required":["category","priority","summary"]}
    """;

    var config = new GenerateContentConfig
    {
        SystemInstruction = new Content { Parts = [new Part { Text = "Classify operational incidents. Do not invent facts." }] },
        Temperature = 0.1,
        ResponseMimeType = "application/json",
        ResponseJsonSchema = JsonNode.Parse(schema)
    };

    var response = await client.Models.GenerateContentAsync(model: model, contents: request.Description, config: config);
    var json = response.Candidates[0].Content.Parts[0].Text ?? throw new InvalidOperationException("No response text.");
    var result = JsonSerializer.Deserialize<AnalysisResult>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
    return Results.Ok(result);
});

app.Run();

public sealed record AnalyzeRequest(string Description);
public sealed record AnalysisResult(string Category, string Priority, string Summary);
