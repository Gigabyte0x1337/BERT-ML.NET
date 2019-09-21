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
                "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.The American Football Conference(AFC) champion Denver Broncos defeated the National Football Conference(NFC) champion Carolina Panthers 24 – 10 to earn their third Super Bowl title.The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
                "Which NFL team represented the AFC at Super Bowl 50?"
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
