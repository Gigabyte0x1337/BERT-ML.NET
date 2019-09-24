using Microsoft.ML;
using ML.BERT.TestApp.Bert;
using ML.BERT.TestApp.Bert.Tokenizers;
using ML.BERT.TestApp.Onnx;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ML.BERT.TestApp
{
    public class BertModel : IDisposable
    {
        private readonly BertModelConfiguration _bertModelConfiguration;
        private List<string> _vocabulary;
        private WordPieceTokenizer _wordPieceTokenizer;
        private PredictionEngine<BertFeature, BertPredictionResult> _predictionEngine;

        public BertModel(BertModelConfiguration bertModelConfiguration)
        {
            _bertModelConfiguration = bertModelConfiguration;
        }

        public void Initialize()
        {
            _vocabulary = ReadVocabularyFile(_bertModelConfiguration.VocabularyFile);
            _wordPieceTokenizer = new WordPieceTokenizer(_vocabulary);

            var onnxModelConfigurator = new OnnxModelConfigurator<BertFeature>(_bertModelConfiguration);
            _predictionEngine = onnxModelConfigurator.GetMlNetPredictionEngine<BertPredictionResult>();
        }

        public List<string> Predict(string context, string question)
        {
            var tokens = _wordPieceTokenizer.Tokenize(question, context);
            var encodedFeature = Encode(tokens);

            var result = _predictionEngine.Predict(encodedFeature);

            var (startIndex, endIndex, probability) = GetBestPredictionFromResult(result);

            return encodedFeature.InputIds
                .Skip(startIndex)
                .Take(endIndex + 1 - startIndex)
                .Select(o => _vocabulary[(int)o])
                .ToList();
        }

        private (int StartIndex, int EndIndex, float Probability) GetBestPredictionFromResult(BertPredictionResult result)
        {
            var bestN = _bertModelConfiguration.BestResultSize;

            var bestStartLogits = result.StartLogits
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(bestN);

            var bestEndLogits = result.EndLogits
                .Select((logit, index) => (Logit: logit, Index: index))
                .OrderByDescending(o => o.Logit)
                .Take(bestN);

            var bestResultsWithScore = bestStartLogits
                .SelectMany(startLogit =>
                    bestEndLogits
                    .Select(endLogit =>
                        (
                            StartLogit: startLogit.Index,
                            EndLogit: endLogit.Index,
                            Score: startLogit.Logit + endLogit.Logit
                        )
                     )
                )
                .Where(entry => !(entry.EndLogit < entry.StartLogit || entry.EndLogit - entry.StartLogit > _bertModelConfiguration.MaxAwnserLength))
                .Take(bestN);

            var (item, probability) = bestResultsWithScore
                .Softmax(o => o.Score)
                .OrderByDescending(o => o.Probability)
                .FirstOrDefault();

            return (StartIndex: item.StartLogit, EndIndex: item.EndLogit, probability);
        }

        private BertFeature Encode(List<(string Token, int Index)> tokens)
        {
            var padding = Enumerable.Repeat(0L, _bertModelConfiguration.MaxSequenceLength - tokens.Count);

            var tokenIndexes = tokens
                .Select(token => (long)token.Index)
                .Concat(padding)
                .ToArray();

            var segmentIndexes = GetSegmentIndexes(tokens)
                .Concat(padding)
                .ToArray();

            var inputMask =
                tokens.Select(o => 1L)
                .Concat(padding)
                .ToArray();

            return new BertFeature()
            {
                InputIds = tokenIndexes,
                SegmentIds = segmentIndexes,
                InputMask = inputMask,
                UniqueIds = new long[] { 0 }
            };
        }

        private IEnumerable<long> GetSegmentIndexes(List<(string token, int index)> tokens)
        {
            var segmentIndex = 0;
            var segmentIndexes = new List<long>();

            foreach (var (token, index) in tokens)
            {
                segmentIndexes.Add(segmentIndex);

                if (token == WordPieceTokenizer.DefaultTokens.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }

        private static List<string> ReadVocabularyFile(string filename)
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

        public void Dispose()
        {
            _predictionEngine.Dispose();
        }
    }
}
