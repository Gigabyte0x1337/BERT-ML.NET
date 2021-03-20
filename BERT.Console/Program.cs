using System;
using System.Text.Json;
using Microsoft.ML.Models.BERT;

namespace BERT.Console
{
    class Program
    {
        static void Main(string[] args)
        {
            var modelConfig = new BertModelConfiguration()
            {
                VocabularyFile = "Model/vocab.txt",
                ModelPath = "Model/bertsquad-10.onnx"
            };

            var model = new BertModel(modelConfig);
            model.Initialize();

            var prediction = model.Predict(args[0],
                args[1]);

            JsonSerializer.Serialize(prediction);
        }
    }
}
