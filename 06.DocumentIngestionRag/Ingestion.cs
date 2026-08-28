using System.Security.Cryptography;
using System.Text;

namespace GenAiLearning.DocumentIngestionRag;

public sealed record SourceDocument(
    string DocumentId,
    string Title,
    string Content,
    string Country,
    string Department,
    string Version,
    string SourceUri);

public sealed record TextChunk(
    string ChunkId,
    string DocumentId,
    string Title,
    string Section,
    int Sequence,
    string Text,
    string Country,
    string Department,
    string Version,
    string SourceUri);

public interface IDocumentChunker
{
    IReadOnlyList<TextChunk> Chunk(SourceDocument document);
}

public sealed class ParagraphDocumentChunker(int maxCharacters = 1500) : IDocumentChunker
{
    private static readonly string[] ParagraphSeparators = ["\r\n\r\n", "\n\n"];

    public IReadOnlyList<TextChunk> Chunk(SourceDocument document)
    {
        var paragraphs = document.Content.Split(
            ParagraphSeparators,
            StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        var chunks = new List<TextChunk>();
        var builder = new StringBuilder();
        var sequence = 0;

        foreach (var paragraph in paragraphs)
        {
            if (builder.Length > 0 && builder.Length + paragraph.Length > maxCharacters)
            {
                AddChunk(builder.ToString());
                builder.Clear();
            }

            if (builder.Length > 0)
            {
                builder.AppendLine().AppendLine();
            }

            builder.Append(paragraph);
        }

        if (builder.Length > 0)
        {
            AddChunk(builder.ToString());
        }

        return chunks;

        void AddChunk(string text)
        {
            var chunkId = $"{document.DocumentId}|{document.Version}|{sequence:D4}";

            chunks.Add(new TextChunk(
                chunkId,
                document.DocumentId,
                document.Title,
                Section: string.Empty,
                sequence++,
                text,
                document.Country,
                document.Department,
                document.Version,
                document.SourceUri));
        }
    }
}

public sealed class DocumentIngestionService(
    IDocumentChunker chunker,
    IEmbeddingService embeddings,
    PostgresVectorStore store)
{
    public async Task<int> IndexAsync(SourceDocument document, CancellationToken cancellationToken = default)
    {
        var chunks = chunker.Chunk(document);

        foreach (var chunk in chunks)
        {
            var vector = await embeddings.GenerateDocumentEmbeddingAsync(
                $"""
                Document: {chunk.Title}
                Section: {chunk.Section}
                {chunk.Text}
                """,
                cancellationToken);

            var id = DeterministicGuid(chunk.ChunkId);

            await store.AddAsync(
                id,
                chunk.DocumentId,
                chunk.Title,
                chunk.Section,
                chunk.Text,
                chunk.Country,
                chunk.Department,
                chunk.Version,
                chunk.SourceUri,
                vector,
                cancellationToken);
        }

        return chunks.Count;
    }

    private static Guid DeterministicGuid(string value)
    {
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(value));

        return new Guid(bytes.AsSpan(0, 16));
    }
}
