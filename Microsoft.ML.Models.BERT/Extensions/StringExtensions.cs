using System.Collections.Generic;

namespace Microsoft.ML.Models.BERT.Extensions
{
    public static class StringExtensions
    {
        public static IEnumerable<string> SplitAndKeep(this string s, params char[] delimiters)
        {
            int start = 0, index;

            while ((index = s.IndexOfAny(delimiters, start)) != -1)
            {
                if (index - start > 0)
                    yield return s.Substring(start, index - start);

                yield return s.Substring(index, 1);

                start = index + 1;
            }

            if (start < s.Length)
            {
                yield return s.Substring(start);
            }
        }
    }
}
