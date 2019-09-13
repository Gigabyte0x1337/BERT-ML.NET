using ML.BERT.TestApp.Onnx;

namespace ML.BERT.TestApp.Bert
{
    public class BertModel : IOnnxModel
    {
        public string ModelPath => "Model/bert.onnx";

        public string[] ModelInput => new string[] { "unique_ids_raw_output___9:0", "segment_ids:0", "input_mask:0", "input_ids:0" };

        public string[] ModelOutput => new string[] { "unstack:1", "unstack:0"/*, "unique_ids:0"*/ };
    }
}
