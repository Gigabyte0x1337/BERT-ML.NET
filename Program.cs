using Microsoft.ML;
using Microsoft.ML.Data;
using ML.BERT.TestApp.Bert;
using ML.BERT.TestApp.Bert.Tokenizers;
using ML.BERT.TestApp.Onnx;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ML.BERT.TestApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var vocabulary = ReadVocabularyFile("Model/vocab.txt").ToList();
            var wordPieceTokenizer = new WordPieceTokenizer(vocabulary);
            var encoder = new BertFeatureEncoder(wordPieceTokenizer);

            var onnxModelConfigurator = new OnnxModelConfigurator<BertFeature>(new BertModel());
            var predictionEngine = onnxModelConfigurator.GetMlNetPredictionEngine<BertPredictionResult>();

            var encodedFeature = encoder.Encode(new string[] {
                "Here is the sentence I want embeddings for.",
                "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
           }, 256);

            var result = predictionEngine.Predict(encodedFeature);

            Console.ReadLine();
        }

        public static IEnumerable<string> ReadVocabularyFile(string filename)
        {
            var vocabulary = new List<string>();

            using (var reader = new StreamReader(filename))
            {
                var line = string.Empty;

                while ((line = reader.ReadLine()) != null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        vocabulary.Add(line);
                    }
                }
            }

            return vocabulary;
        }
    }
}
