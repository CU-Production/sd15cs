using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace StableDiffusion;

public static class TextProcessing
{
    public static DenseTensor<float> PreprocessText(String prompt, StableDiffusionConfig config)
    {
        var textTokenized = TokenizeText(prompt, config);
        var textPromptEmbeddings = TextEncoder(textTokenized, config).ToArray();

        var uncondInputTokens = CreateUncondInput();
        var uncondEmbedding = TextEncoder(uncondInputTokens, config).ToArray();

        DenseTensor<float> textEmbeddings = new DenseTensor<float>(new int[] { 2, 77, 768 });

        for (int i = 0; i < textPromptEmbeddings.Length; ++i)
        {
            textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];
            textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];
        }

        return textEmbeddings;
    }

    public static int[] CreateUncondInput()
    {
        Int32 blankTokenValue = 49407;
        Int32 modelMaxLength = 77;
        var inputIds = new List<Int32>();
        inputIds.Add(49406);
        var pad = Enumerable.Repeat(blankTokenValue, modelMaxLength - inputIds.Count).ToArray();
        inputIds.AddRange(pad);

        return inputIds.ToArray();
    }

    public static int[] TokenizeText(string text, StableDiffusionConfig config)
    {
        var sessionOptions = new SessionOptions();
        // sessionOptions.RegisterCustomOpLibraryV2(config.OrtExtensionsPath, out var libraryHandle);
        sessionOptions.RegisterOrtExtensions();

        var tokenizeSession = new InferenceSession(config.TokenizerOnnxPath, sessionOptions);
        var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
        var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };

        var tokens = tokenizeSession.Run(inputString);
        
        var inputIds = (tokens.ToList().First().Value as IEnumerable<long>).ToArray();
        Console.WriteLine(String.Join(" ", inputIds));

        // Cast inputIds to Int32
        var InputIdsInt32 = inputIds.Select(x => (int)x).ToArray();
        
        var modelMaxLength = 77;
        // Pad array with 49407 until length is modelMaxLength
        if (InputIdsInt32.Length < modelMaxLength)
        {
            var pad = Enumerable.Repeat(49407, 77 - InputIdsInt32.Length).ToArray();
            InputIdsInt32 = InputIdsInt32.Concat(pad).ToArray();
        }

        return InputIdsInt32;
    }

    public static DenseTensor<float> TextEncoder(int[] tokenizedInput, StableDiffusionConfig config)
    {
        var input_ids = TensorHelper.CreateTensor(tokenizedInput, new int[] { 1, tokenizedInput.Count() });

        var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<int>("input_ids", input_ids) };
        
        // DirectML EP
        var sessionOptions = config.GetSessionOptionsForEp();

        var encodeSession = new InferenceSession(config.TextEncoderOnnxPath, sessionOptions);

        var encoded = encodeSession.Run(input);

        var lastHiddenState = (encoded.ToList().First().Value as IEnumerable<float>).ToArray();
        var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState.ToArray(), new int[] { 1, 77, 768 });

        return lastHiddenStateTensor;
    }
}