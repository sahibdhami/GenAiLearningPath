namespace GenAiLearning.IntroducingRag;

public sealed record DocumentChunk(string Id, string Title, string Text, float[] Embedding);

public sealed record SearchHit(DocumentChunk Chunk, double Similarity);

public static class VectorMath
{
    public static double CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Vector dimensions differ.");
        }

        double dot = 0;
        double magnitudeA = 0;
        double magnitudeB = 0;

        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }

        return magnitudeA == 0 || magnitudeB == 0
            ? 0
            : dot / (Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB));
    }
}
