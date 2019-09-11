using Microsoft.ML;
using System.Collections.Generic;

namespace ML.BERT.TestApp.Onnx
{
    public class OnnxModelConfigurator<TFeature> where TFeature : class
    {
        private readonly MLContext mlContext;
        private readonly ITransformer mlModel;

        public OnnxModelConfigurator(IOnnxModel onnxModel)
        {
            mlContext = new MLContext();

            mlModel = SetupMlNetModel(onnxModel);
        }

        private ITransformer SetupMlNetModel(IOnnxModel onnxModel)
        {
            var dataView = mlContext.Data
                .LoadFromEnumerable(new List<TFeature>());

            var pipeline = mlContext.Transforms
                            .ApplyOnnxModel(modelFile: onnxModel.ModelPath, outputColumnNames: onnxModel.ModelOutput, inputColumnNames: onnxModel.ModelInput);

            var mlNetModel = pipeline.Fit(dataView);

            return mlNetModel;
        }

        public PredictionEngine<TFeature, T> GetMlNetPredictionEngine<T>()
            where T : class, new()
        {
            return mlContext.Model.CreatePredictionEngine<TFeature, T>(mlModel);
        }

        public void SaveMLNetModel(string mlnetModelFilePath)
        {
            // Save/persist the model to a .ZIP file to be loaded by the PredictionEnginePool
            mlContext.Model.Save(mlModel, null, mlnetModelFilePath);
        }
    }
}
