using Microsoft.ML;
using System.Collections.Generic;

namespace ML.BERT.TestApp.Onnx
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
            var dataView = _mlContext.Data
                .LoadFromEnumerable(new List<TFeature>());

            var pipeline = _mlContext.Transforms
                            .ApplyOnnxModel(modelFile: onnxModel.ModelPath, outputColumnNames: onnxModel.ModelOutput, inputColumnNames: onnxModel.ModelInput);

            var mlNetModel = pipeline.Fit(dataView);

            return mlNetModel;
        }

        public PredictionEngine<TFeature, T> GetMlNetPredictionEngine<T>()
            where T : class, new()
        {
            return _mlContext.Model.CreatePredictionEngine<TFeature, T>(_mlModel);
        }

        public void SaveMLNetModel(string mlnetModelFilePath)
        {
            // Save/persist the model to a .ZIP file to be loaded by the PredictionEnginePool
            _mlContext.Model.Save(_mlModel, null, mlnetModelFilePath);
        }
    }
}
