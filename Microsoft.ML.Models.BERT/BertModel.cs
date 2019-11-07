using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Models.BERT.Extensions;
using Microsoft.ML.Models.BERT.Input;
using Microsoft.ML.Models.BERT.Onnx;
using Microsoft.ML.Models.BERT.Output;
using Microsoft.ML.Models.BERT.Tokenizers;

namespace Microsoft.ML.Models.BERT
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

        public (List<string> tokens, float probability) Predict(string context, string question)
        {
            var tokens = _wordPieceTokenizer.Tokenize(question, context);
            var encodedFeature = Encode(tokens);

            var result = _predictionEngine.Predict(encodedFeature);
            var contextStart = tokens.FindIndex(o => o.Token == WordPieceTokenizer.DefaultTokens.Separation);

            var (startIndex, endIndex, probability) = GetBestPredictionFromResult(result, contextStart);

            var predictedTokens = encodedFeature.InputIds
                .Skip(startIndex)
                .Take(endIndex + 1 - startIndex)
                .Select(o => _vocabulary[(int)o])
                .ToList();

            var stichedTokens = StitchSentenceBackTogether(predictedTokens);

            return (stichedTokens, probability);
        }

        private List<string> StitchSentenceBackTogether(List<string> tokens)
        {
            var currentToken = string.Empty;

            tokens.Reverse();

            var tokensStitched = new List<string>();

            foreach (var token in tokens)
            {
                if (!token.StartsWith("##"))
                {
                    currentToken = token + currentToken;
                    tokensStitched.Add(currentToken);
                    currentToken = string.Empty;
                } else
                {
                    currentToken = token.Replace("##", "") + currentToken;
                }
            }

            tokensStitched.Reverse();

            return tokensStitched;
        }

        private (int StartIndex, int EndIndex, float Probability) GetBestPredictionFromResult(BertPredictionResult result, int minIndex)
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
                .Where(entry => !(entry.EndLogit < entry.StartLogit || entry.EndLogit - entry.StartLogit > _bertModelConfiguration.MaxAnswerLength || entry.StartLogit == 0 && entry.EndLogit == 0 || entry.StartLogit < minIndex))
                .Take(bestN);

            var (item, probability) = bestResultsWithScore
                .Softmax(o => o.Score)
                .OrderByDescending(o => o.Probability)
                .FirstOrDefault();

            return (StartIndex: item.StartLogit, EndIndex: item.EndLogit, probability);
        }

        private BertFeature Encode(List<(string Token, int Index)> tokens)
        {
            var padding = Enumerable
                .Repeat(0L, _bertModelConfiguration.MaxSequenceLength - tokens.Count)
                .ToList();

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
                string line;

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
