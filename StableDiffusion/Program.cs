using SixLabors.ImageSharp;

namespace StableDiffusion
{
    public class Program
    {
        static void Main(string[] args)
        {
            System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();

            //Default args
            var prompt = "a fireplace in an old cabin in the woods";
            Console.WriteLine(prompt);

            var config = new StableDiffusionConfig();
            config.NumInferenceSteps = 15;
            config.GuidanceScale = 7.5;
            config.ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.DirectML;
            config.DeviceId = 0; // on notebook, may have more than one gpu, and discrete gpu may be 1, 0 is integrated GPU

            config.TextEncoderOnnxPath = @"G:\wrokspace2\ML\stable-diffusion-v1-5\text_encoder\model.onnx";
            config.UnetOnnxPath        = @"G:\wrokspace2\ML\stable-diffusion-v1-5\unet\model.onnx";
            config.VaeDecoderOnnxPath  = @"G:\wrokspace2\ML\stable-diffusion-v1-5\vae_decoder\model.onnx";
            config.SafetyModelPath     = @"G:\wrokspace2\ML\stable-diffusion-v1-5\safety_checker\model.onnx";
            
            // Inference Stable Diff
            var image = UNet.Inference(prompt, config);
            
            // If image failed or was unsafe it will return null.
            if (image == null)
            {
                Console.WriteLine("Unable to create image, please try again.");
            }
            
            // save image
            var imageName = $"sd_image_{DateTime.Now.ToString("yyyyMMddHHmmssfff")}.png";
            var imagePath = Path.Combine(Directory.GetCurrentDirectory(), config.ImageOutputPath, imageName);
            
            image.Save(imagePath);
            
            Console.WriteLine($"Image saved to: {imagePath}");
            
            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Time taken: " + elapsedMs + "ms");
        }
    }
}
