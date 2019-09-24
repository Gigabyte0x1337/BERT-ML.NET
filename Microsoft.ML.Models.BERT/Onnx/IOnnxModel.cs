namespace Microsoft.ML.Models.BERT.Onnx
{
    public interface IOnnxModel
    {
        string ModelPath { get; }
        string[] ModelInput { get; }
        string[] ModelOutput { get; }
    }
}
