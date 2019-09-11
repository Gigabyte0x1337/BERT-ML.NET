using ML.BERT.TestApp.Bert.Tokenizers;
using System.Collections.Generic;
using System.Linq;

namespace ML.BERT.TestApp.Bert
{
    public class BertFeatureEncoder
    {
        private readonly WordPieceTokenizer _tokenizer;

        public BertFeatureEncoder(WordPieceTokenizer tokenizer)
        {
            _tokenizer = tokenizer;
        }

        public BertFeature Encode(string[] texts, int sequenceLength)
        {
            var tokens = _tokenizer.Tokenize(texts)
                .ToList();

            var padding = Enumerable.Repeat((long)0, sequenceLength - tokens.Count);

            var uniqueTokenIndexes = Enumerable.Range(1000000000, sequenceLength)
                .Select(o => (long)o)
                .ToArray();

            var tokenIndexes = tokens
                .Select(o => (long)o.Item2)
                .Concat(padding)
                .ToArray();

            var segmentIndexes = GetSegmentIndexes(tokens)
                .Concat(padding)
                .ToArray();

            var inputMask =
                tokens.Select(o => (long)1)
                .Concat(padding)
                .ToArray();

            return new BertFeature()
            {
                InputIds = tokenIndexes,
                SegmentIds = segmentIndexes,
                InputMask = inputMask,
                UniqueIds = uniqueTokenIndexes
            };
        }

        private IEnumerable<long> GetSegmentIndexes(List<(string, int)> tokens)
        {
            var segmentIndexes = new List<long>();
            var segmentIndex = 0;
            for (int i = 0; i < tokens.Count(); i++)
            {
                segmentIndexes.Add(segmentIndex);

                if (tokens[i].Item1 == WordPieceTokenizer.DefaultTokens.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }
    }
}
