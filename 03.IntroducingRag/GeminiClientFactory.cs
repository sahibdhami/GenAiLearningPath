using Google.GenAI;

namespace GenAiLearning.IntroducingRag;

public static class GeminiClientFactory
{
    public static Client Create(GoogleCloudOptions options) =>
        new(project: options.ProjectId, location: options.Location, enterprise: true);
}
