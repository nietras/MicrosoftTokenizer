// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Unicode;
using System.Threading.Tasks;
using Microsoft.DeepDev;
using Microsoft.DeepDev.TokenizerLib.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.VisualStudio.TestTools.UnitTesting.Logging;
using Newtonsoft.Json;

namespace TokenizerTest
{
    [TestClass]
    public class TikTokenizerUnitTest
    {
        private ITokenizer Tokenizer;
        private ITokenizer Tokenizer_gpt2;
        private ITokenizer Tokenizer_p50k_base;
        private ITokenizer Tokenizer_r50k_base;
        private ITokenizer Tokenizer_p50k_edit;

        const string IM_START = "<|im_start|>";
        const string IM_END = "<|im_end|>";

        static readonly Dictionary<string, int> SpecialTokens = new Dictionary<string, int>{
                                                    { IM_START, 100264},
                                                    { IM_END, 100265},
                                                };

        [TestInitialize]
        public async Task TikTokenizerUnitTestInitialize()
        {
            Tokenizer = await TokenizerBuilder.CreateByModelNameAsync("gpt-4", SpecialTokens);
            Tokenizer_gpt2 = await TokenizerBuilder.CreateByEncoderNameAsync("gpt2");
            Tokenizer_p50k_base = await TokenizerBuilder.CreateByEncoderNameAsync("p50k_base");
            Tokenizer_r50k_base = await TokenizerBuilder.CreateByEncoderNameAsync("r50k_base");
            Tokenizer_p50k_edit = await TokenizerBuilder.CreateByEncoderNameAsync("p50k_edit");
        }

        [TestMethod]
        public async Task DownloadAndStats()
        {
            foreach (var (name, url) in TokenizerBuilder.NameToVocabularyUrl)
            {
                var fileName = name + ".tiktoken";
                if (!File.Exists(fileName))
                {
                    using var httpClient = new HttpClient();
                    using var urlStream = await httpClient.GetStreamAsync(url);
                    using var localStream = new FileStream(fileName, FileMode.Create, FileAccess.Write);
                    urlStream.CopyTo(localStream);
                    Log($"Downloaded {fileName}");
                }
                using var stream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
                var vocabulary = TikTokenizer.LoadTikTokenBpe(stream);
                VocabularyStats(fileName, vocabulary);
            }
        }

        static readonly Action<string> Log = t => { Trace.WriteLine(t); };

        record Token(byte[] Bytes, string Text, int Id);
        record TokenAccum(string Text) // Text may be substitute if "invalid"/incomplete utf8 sequence
        {
            public List<Token> Tokens { get; } = new();
        }

        private void VocabularyStats(string fileName, Dictionary<byte[], int> vocabulary)
        {
            var bytesToTokenAccum = new Dictionary<byte[], TokenAccum>(vocabulary.Count, new ByteArrayComparer());
            var chars = new char[2048];
            foreach (var (bytes, token) in vocabulary)
            {
                // check if bytes is a valid and full utf8 sequence
                var status = Utf8.ToUtf16(bytes, chars, out var bytesRead, out var charsWritten, replaceInvalidSequences: false);
                if (status != OperationStatus.Done || charsWritten <= 0)
                {
                    //Log($"Invalid utf8 sequence: {BitConverter.ToString(bytes)}");
                    // If only 1 byte then might be partial utf8 sequence
                    // This needs special handling, since such a byte cannot be converted to a string
                    //if (bytes.Length == 1)
                    //{
                    var byteAsText = $"!<{BitConverter.ToString(bytes)}>!";
                    var byteAccum = new TokenAccum(byteAsText);
                    byteAccum.Tokens.Add(new(bytes, byteAsText, token));
                    bytesToTokenAccum.Add(bytes, byteAccum);
                }
                // Really what needs to be handled is whether utf8 byte sequence is valid/full etc.
                else
                {
                    var span = chars.AsSpan()[..charsWritten];
                    var textOriginal = new string(span);

                    var trimmed = span.Trim(' ');
                    var maybeTrimmed = trimmed.Length > 0 ? trimmed : span;
                    var textNormalized = new string(maybeTrimmed).ToLowerInvariant();
                    var bytesNormalized = Encoding.UTF8.GetBytes(textNormalized);
                    if (!bytesToTokenAccum.TryGetValue(bytesNormalized, out var accum))
                    {
                        accum = new(textNormalized);
                        bytesToTokenAccum.Add(bytesNormalized, accum);
                    }
                    accum.Tokens.Add(new(bytes, textOriginal, token));
                }
            }
            Log($"{fileName} {vocabulary.Count} tokens {bytesToTokenAccum.Count} tokens if ignore case and trim spaces ({1 - (bytesToTokenAccum.Count / (double)vocabulary.Count):P2} less)");

            var sorted = bytesToTokenAccum.OrderByDescending(kv => kv.Value.Tokens.Count).ToList();

            Log("REPEATED TOKENS");
            foreach (var (normText, accum) in sorted.Take(20))
            {
                Log($"'{EscapeSpecialWhiteSpace(accum.Text)}' Count {accum.Tokens.Count} Tokens: " +
                    $"{string.Join(", ", accum.Tokens.Select(t => $"'{EscapeSpecialWhiteSpace(t.Text)}':0x{BitConverter.ToString(t.Bytes)}:{t.Id}"))}");
            }

            Log("LONG TOKENS");
            vocabulary.OrderByDescending(p => p.Key.Length).Take(20).ToList().ForEach(kv =>
            {
                var (bytes, token) = kv;
                var text = Encoding.UTF8.GetString(bytes);
                Log($"'{EscapeSpecialWhiteSpace(text)}' Length {bytes.Length} Bytes: {BitConverter.ToString(bytes)}");
            });

            static string EscapeSpecialWhiteSpace(string t) =>
                t.Replace("\t", "\\t").Replace("\r", "\\r").Replace("\n", "\\n");
        }

        [TestMethod]
        public void TestEncode0()
        {
            var text = "Hello World";
            var encoded = Tokenizer.Encode(text, new HashSet<string>(SpecialTokens.Keys));
            Assert.AreEqual(2, encoded.Count);
            Assert.AreEqual(9906, encoded[0]);
            Assert.AreEqual(4435, encoded[1]);
            var decoded = Tokenizer.Decode(encoded.ToArray());
            Assert.AreEqual(text, decoded);
        }


        [TestMethod]
        public void TestEncode1()
        {
            var text = "<|im_start|>Hello World<|im_end|>";
            var encoded = Tokenizer.Encode(text);
            Assert.AreEqual(4, encoded.Count);
            Assert.AreEqual(100264, encoded[0]);
            Assert.AreEqual(9906, encoded[1]);
            Assert.AreEqual(4435, encoded[2]);
            Assert.AreEqual(100265, encoded[3]);
            var decoded = Tokenizer.Decode(encoded.ToArray());
            Assert.AreEqual(text, decoded);
        }

        [TestMethod]
        public void TestEncode2()
        {
            var text = File.ReadAllText("./testData/lib.rs.txt");
            var encoded = Tokenizer.Encode(text, new HashSet<string>(SpecialTokens.Keys));
            Assert.AreEqual(5584, encoded.Count);

            encoded = Tokenizer.Encode(text, false);
            Assert.AreEqual(5584, encoded.Count);

            string json = File.ReadAllText("./testData/tokens.json");
            var expected = JsonConvert.DeserializeObject<int[]>(json);

            for (int i = 0; i < encoded.Count; i++)
            {
                Assert.AreEqual(expected[i], encoded[i]);
            }
            Assert.AreEqual(expected.Length, encoded.Count);

            var decoded = Tokenizer.Decode(encoded.ToArray());
            Assert.AreEqual(text, decoded);
        }

        [TestMethod]
        public void TestEncode3()
        {
            var text = "<|im_start|>Hello<|im_end|> World";
            var encoded = Tokenizer.Encode(text, new HashSet<string>(SpecialTokens.Keys));
            Assert.AreEqual(4, encoded.Count);
            Assert.AreEqual(100264, encoded[0]);
            Assert.AreEqual(9906, encoded[1]);
            Assert.AreEqual(100265, encoded[2]);
            Assert.AreEqual(4435, encoded[3]);
            var decoded = Tokenizer.Decode(encoded.ToArray());
            Assert.AreEqual(text, decoded);
        }

        [TestMethod]
        public void TestEncode4()
        {
            var text = "";
            var encoded = Tokenizer.Encode(text, new HashSet<string>(SpecialTokens.Keys));
            Assert.AreEqual(0, encoded.Count);
        }


        [TestMethod]
        public void TestEncode5()
        {
            var text = "<|im_start|>Hello ⭐ World<|im_end|>";
            var encoded = Tokenizer.Encode(text, new HashSet<string>(SpecialTokens.Keys));
            Assert.AreEqual(6, encoded.Count);
            Assert.AreEqual(100264, encoded[0]);
            Assert.AreEqual(9906, encoded[1]);
            Assert.AreEqual(2928, encoded[2]);
            Assert.AreEqual(99834, encoded[3]);
            Assert.AreEqual(4435, encoded[4]);
            Assert.AreEqual(100265, encoded[5]);
            var decoded = Tokenizer.Decode(encoded.ToArray());
            Assert.AreEqual(text, decoded);
        }

        [TestMethod]
        public void TestEncodeTrimSuffix()
        {
            var text = "<|im_start|>Hello World<|im_end|>";
            var encodedText = "<|im_start|>Hello World";
            var encoded = Tokenizer.EncodeTrimSuffix(text, new HashSet<string>(SpecialTokens.Keys), 4);
            Assert.AreEqual(4, encoded.TokenIds.Count);
            Assert.AreEqual(text, encoded.Text);

            encoded = Tokenizer.EncodeTrimSuffix(text, 4, false);
            Assert.AreEqual(4, encoded.TokenIds.Count);
            Assert.AreEqual("<|im_start", encoded.Text);

            encoded = Tokenizer.EncodeTrimSuffix(text, 4);
            Assert.AreEqual(4, encoded.TokenIds.Count);
            Assert.AreEqual(text, encoded.Text);

            encoded = Tokenizer.EncodeTrimSuffix(text, new HashSet<string>(SpecialTokens.Keys), 5);
            Assert.AreEqual(4, encoded.TokenIds.Count);
            Assert.AreEqual(text, encoded.Text);

            encoded = Tokenizer.EncodeTrimSuffix(text, new HashSet<string>(SpecialTokens.Keys), 3);
            Assert.AreEqual(3, encoded.TokenIds.Count);
            Assert.AreEqual(encodedText, encoded.Text);
            var decoded = Tokenizer.Decode(encoded.TokenIds.ToArray());
            Assert.AreEqual(encodedText, decoded);
        }

        [TestMethod]
        public void TestEncodeTrimSuffix2()
        {
            var text = "<|im_start|>Hello TempWorld<|im_end|>";
            var encodedText = "<|im_start|>Hello";
            var encoded = Tokenizer.EncodeTrimSuffix(text, new HashSet<string>(SpecialTokens.Keys), 5);
            Assert.AreEqual(5, encoded.TokenIds.Count);
            Assert.AreEqual(text, encoded.Text);

            encoded = Tokenizer.EncodeTrimSuffix(text, new HashSet<string>(SpecialTokens.Keys), 6);
            Assert.AreEqual(5, encoded.TokenIds.Count);
            Assert.AreEqual(text, encoded.Text);

            encoded = Tokenizer.EncodeTrimSuffix(text, new HashSet<string>(SpecialTokens.Keys), 3);
            Assert.AreEqual(2, encoded.TokenIds.Count);
            Assert.AreEqual(encodedText, encoded.Text);
            var decoded = Tokenizer.Decode(encoded.TokenIds.ToArray());
            Assert.AreEqual(encodedText, decoded);
        }



        [TestMethod]
        public void TestEncodeTrimPrefix()
        {
            var text = "<|im_start|>Hello World<|im_end|>";
            var encodedText = "Hello World<|im_end|>";
            var encoded = Tokenizer.EncodeTrimPrefix(text, new HashSet<string>(SpecialTokens.Keys), 4);
            Assert.AreEqual(4, encoded.TokenIds.Count);
            Assert.AreEqual(text, encoded.Text);

            encoded = Tokenizer.EncodeTrimPrefix(text, 4, false);
            Assert.AreEqual(4, encoded.TokenIds.Count);
            Assert.AreEqual("im_end|>", encoded.Text);

            encoded = Tokenizer.EncodeTrimPrefix(text, 4);
            Assert.AreEqual(4, encoded.TokenIds.Count);
            Assert.AreEqual(text, encoded.Text);

            encoded = Tokenizer.EncodeTrimPrefix(text, new HashSet<string>(SpecialTokens.Keys), 5);
            Assert.AreEqual(4, encoded.TokenIds.Count);
            Assert.AreEqual(text, encoded.Text);

            encoded = Tokenizer.EncodeTrimPrefix(text, new HashSet<string>(SpecialTokens.Keys), 3);
            Assert.AreEqual(3, encoded.TokenIds.Count);
            Assert.AreEqual(encodedText, encoded.Text);
            var decoded = Tokenizer.Decode(encoded.TokenIds.ToArray());
            Assert.AreEqual(encodedText, decoded);
        }


        [TestMethod]
        public void TestEncodeTrimPrefix2()
        {
            var text = "<|im_start|>HelloTemp World<|im_end|>";
            var encodedText = " World<|im_end|>";
            var encoded = Tokenizer.EncodeTrimPrefix(text, new HashSet<string>(SpecialTokens.Keys), 5);
            Assert.AreEqual(5, encoded.TokenIds.Count);
            Assert.AreEqual(text, encoded.Text);

            encoded = Tokenizer.EncodeTrimPrefix(text, new HashSet<string>(SpecialTokens.Keys), 6);
            Assert.AreEqual(5, encoded.TokenIds.Count);
            Assert.AreEqual(text, encoded.Text);

            encoded = Tokenizer.EncodeTrimPrefix(text, new HashSet<string>(SpecialTokens.Keys), 3);
            Assert.AreEqual(2, encoded.TokenIds.Count);
            Assert.AreEqual(encodedText, encoded.Text);
            var decoded = Tokenizer.Decode(encoded.TokenIds.ToArray());
            Assert.AreEqual(encodedText, decoded);
        }

        [TestMethod]
        public void TestEncodeGpt2()
        {
            var text = File.ReadAllText("./testData/lib.rs.txt");
            var encoded = Tokenizer_gpt2.Encode(text, new HashSet<string>());
            Assert.AreEqual(11378, encoded.Count);

            string json = File.ReadAllText("./testData/tokens_gpt2.json");
            var expected = JsonConvert.DeserializeObject<int[]>(json);

            for (int i = 0; i < encoded.Count; i++)
            {
                Assert.AreEqual(expected[i], encoded[i]);
            }
            Assert.AreEqual(expected.Length, encoded.Count);

            var decoded = Tokenizer_gpt2.Decode(encoded.ToArray());
            Assert.AreEqual(text, decoded);
        }

        [TestMethod]
        public void TestEncodeP50kbase()
        {
            var text = File.ReadAllText("./testData/lib.rs.txt");
            var encoded = Tokenizer_p50k_base.Encode(text, new HashSet<string>());
            Assert.AreEqual(7230, encoded.Count);

            string json = File.ReadAllText("./testData/tokens_p50k_base.json");
            var expected = JsonConvert.DeserializeObject<int[]>(json);

            for (int i = 0; i < encoded.Count; i++)
            {
                Assert.AreEqual(expected[i], encoded[i]);
            }
            Assert.AreEqual(expected.Length, encoded.Count);

            var decoded = Tokenizer_p50k_base.Decode(encoded.ToArray());
            Assert.AreEqual(text, decoded);
        }

        [TestMethod]
        public void TestEncodeP50kedit()
        {
            var text = File.ReadAllText("./testData/lib.rs.txt");
            var encoded = Tokenizer_p50k_edit.Encode(text, new HashSet<string>());
            Assert.AreEqual(7230, encoded.Count);

            string json = File.ReadAllText("./testData/tokens_p50k_edit.json");
            var expected = JsonConvert.DeserializeObject<int[]>(json);

            for (int i = 0; i < encoded.Count; i++)
            {
                Assert.AreEqual(expected[i], encoded[i]);
            }
            Assert.AreEqual(expected.Length, encoded.Count);

            var decoded = Tokenizer_p50k_edit.Decode(encoded.ToArray());
            Assert.AreEqual(text, decoded);
        }

        [TestMethod]
        public void TestEncodeR50kbase()
        {
            var text = File.ReadAllText("./testData/lib.rs.txt");
            var encoded = Tokenizer_r50k_base.Encode(text, new HashSet<string>());
            Assert.AreEqual(11378, encoded.Count);

            string json = File.ReadAllText("./testData/tokens_r50k_base.json");
            var expected = JsonConvert.DeserializeObject<int[]>(json);

            for (int i = 0; i < encoded.Count; i++)
            {
                Assert.AreEqual(expected[i], encoded[i]);
            }
            Assert.AreEqual(expected.Length, encoded.Count);

            var decoded = Tokenizer_r50k_base.Decode(encoded.ToArray());
            Assert.AreEqual(text, decoded);
        }

    }
}
