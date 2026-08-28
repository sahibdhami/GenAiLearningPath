using Google.GenAI;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

var projectId = builder.Configuration["GoogleCloud:ProjectId"]
    ?? throw new InvalidOperationException("Set GoogleCloud:ProjectId in appsettings.json.");
var location = builder.Configuration["GoogleCloud:Location"] ?? "global";
var model = builder.Configuration["GoogleCloud:GeminiModel"] ?? "gemini-3.7-flash";
var client = new Client(project: projectId, location: location, enterprise: true);

app.MapGet("/ask", async (string q) =>
{
    var response = await client.Models.GenerateContentAsync(model: model, contents: q);
    return Results.Ok(new { answer = response.Candidates[0].Content.Parts[0].Text });
});

app.Run();
