using Microsoft.ML;
using Microsoft.ML.Models.BERT.Onnx;
using System.Collections.Generic;

namespace Microsoft.ML.Models.BERT.Onnx
{
    public class OnnxModelConfigurator<TFeature> where TFeature : class
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _mlModel;

        public OnnxModelConfigurator(IOnnxModel onnxModel)
        {
            _mlContext = new MLContext();
            _mlModel = SetupMlNetModel(onnxModel);
        }

        private ITransformer SetupMlNetModel(IOnnxModel onnxModel)
        {
            bool hasGpu = false;

            var dataView = _mlContext.Data
                .LoadFromEnumerable(new List<TFeature>());

            var pipeline = _mlContext.Transforms
                            .ApplyOnnxModel(modelFile: onnxModel.ModelPath, outputColumnNames: onnxModel.ModelOutput, inputColumnNames: onnxModel.ModelInput, gpuDeviceId: hasGpu ? 0 : (int?)null);

            var mlNetModel = pipeline.Fit(dataView);

            return mlNetModel;
        }

        public PredictionEngine<TFeature, T> GetMlNetPredictionEngine<T>() where T : class, new()
        {
            return _mlContext.Model.CreatePredictionEngine<TFeature, T>(_mlModel);
        }

        public void SaveMLNetModel(string mlnetModelFilePath)
        {
            _mlContext.Model.Save(_mlModel, null, mlnetModelFilePath);
        }
    }
}
