using System;
using System.Text.Json;
using Microsoft.ML.Models.BERT;

var modelConfig = new BertModelConfiguration()
{
    VocabularyFile = "Model/vocab.txt",
    ModelPath = "Model/bertsquad-10.onnx"
};

var model = new BertModel(modelConfig);
model.Initialize();

var (tokens, probability) = model.Predict(args[0], args[1]);
Console.WriteLine(JsonSerializer.Serialize(new
{
    Probability = probability,
    Tokens = tokens
}));
