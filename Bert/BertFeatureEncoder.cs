﻿using ML.BERT.TestApp.Bert.Tokenizers;
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

            var padding = Enumerable.Repeat(0L, sequenceLength - tokens.Count);

            var tokenIndexes = tokens
                .Select(token => (long)token.VocabularyIndex)
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
    }
}
