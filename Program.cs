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

            var question = new string[] {
                "How many tons of dust remains in the air?",
                "NASA's CALIPSO satellite has measured the amount of dust transported by wind from the Sahara to the Amazon: an average 182 million tons of dust are windblown out of the Sahara each year, at 15 degrees west longitude, across 1,600 miles (2,600 km) over the Atlantic Ocean (some dust falls into the Atlantic), then at 35 degrees West longitude at the eastern coast of South America, 27.7 million tons (15%) of dust fall over the Amazon basin, 132 million tons of dust remain in the air, 43 million tons of dust are windblown and falls on the Caribbean Sea, past 75 degrees west longitude.",
            };

            var encodedFeature = encoder.Encode(question, 256);

            var result = predictionEngine.Predict(encodedFeature);

            var maxSequenceLength = 30;
            var bestN = 20;

            var bestStartLogits = result.StartLogits
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(bestN);

            var bestEndLogits = result.EndLogits
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(bestN);

            var bestResults = bestStartLogits
                .SelectMany(startLogit =>
                    bestEndLogits
                    .Select(endLogit =>
                        (
                            StartLogit: startLogit,
                            EndLogit: endLogit,
                            Score: startLogit.Logit + endLogit.Logit
                        )
                     )
                )
                .Where(entry => !(entry.EndLogit.Index < entry.StartLogit.Index || entry.EndLogit.Index - entry.StartLogit.Index > maxSequenceLength))
                .Take(bestN);

            var (item, probability) = bestResults
                .Softmax(o => o.Score)
                .OrderByDescending(o => o.Probability)
                .FirstOrDefault();

            var tokens = encodedFeature.InputIds
                .Where((_, index) => index >= item.StartLogit.Index && index <= item.EndLogit.Index)
                .Select(o => vocabulary[(int)o])
                .ToList();

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
