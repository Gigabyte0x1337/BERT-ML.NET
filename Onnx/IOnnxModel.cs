namespace ML.BERT.TestApp.Onnx
{
    public interface IOnnxModel
    {
        string ModelPath { get; }
        string[] ModelInput { get; }
        string[] ModelOutput { get; }
    }
}
