using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace StableDiffusion;

public static class VaeDecoder
{
    public static Tensor<float> Decoder(List<NamedOnnxValue> input, string VaeDecoderOnnxPath)
    {
        var vaeDecodeSession = new InferenceSession(VaeDecoderOnnxPath);

        var output = vaeDecodeSession.Run(input);
        var result = (output.ToList().First().Value as Tensor<float>);

        return result;
    }

    public static Image<Rgba32> ConvertToImage(Tensor<float> output, StableDiffusionConfig config, int width = 512,
        int height = 512)
    {
        var result = new Image<Rgba32>(width, height);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                result[x, y] = new Rgba32(
                    (byte)(Math.Round(Math.Clamp((output[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                    (byte)(Math.Round(Math.Clamp((output[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                    (byte)(Math.Round(Math.Clamp((output[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
                    );
            }
        }

        return result;
    }
}