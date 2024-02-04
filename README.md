# sd15cs
Stable Diffusion with C# and ONNX Runtime

## how to use

1. download sd1.5 onnx models

```bash
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 -b onnx
```

2. change config in program.cs

```csharp
// program.cs
config.TextEncoderOnnxPath = @"G:\wrokspace2\ML\stable-diffusion-v1-5\text_encoder\model.onnx";
config.UnetOnnxPath        = @"G:\wrokspace2\ML\stable-diffusion-v1-5\unet\model.onnx";
config.VaeDecoderOnnxPath  = @"G:\wrokspace2\ML\stable-diffusion-v1-5\vae_decoder\model.onnx";
config.SafetyModelPath     = @"G:\wrokspace2\ML\stable-diffusion-v1-5\safety_checker\model.onnx";
```

3. change prompt and run

```csharp
// program.cs
var prompt = "a fireplace in an old cabin in the woods";
```

## references
- https://github.com/cassiebreviu/StableDiffusion
- https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html
- https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/onnx
- https://github.com/axodox/axodox-machinelearning
