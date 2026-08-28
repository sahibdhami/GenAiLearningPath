namespace GenAiLearning.DocumentIngestionRag;

public sealed class GoogleCloudOptions
{
    public string ProjectId { get; init; } = "";
    public string Location { get; init; } = "global";
    public string GeminiModel { get; init; } = "gemini-2.5-flash";
    public string EmbeddingModel { get; init; } = "gemini-embedding-001";
    public int EmbeddingDimensions { get; init; } = 768;
}
