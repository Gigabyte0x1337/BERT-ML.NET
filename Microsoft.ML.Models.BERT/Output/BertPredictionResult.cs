using Microsoft.ML.Data;

namespace Microsoft.ML.Models.BERT.Output
{
    internal class BertPredictionResult
    {
        [VectorType(1, 256)]
        [ColumnName("unstack:1")]
        public float[] EndLogits { get; set; }

        [VectorType(1, 256)]
        [ColumnName("unstack:0")]
        public float[] StartLogits { get; set; }

        [VectorType(1)]
        [ColumnName("unique_ids:0")]
        public long[] UniqueIds { get; set; }
    }
}
