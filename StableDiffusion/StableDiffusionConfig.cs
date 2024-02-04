using Microsoft.ML.OnnxRuntime;

namespace StableDiffusion;

public class StableDiffusionConfig
{
    public enum ExecutionProvider
    {
        DirectML = 0,
        Cuda = 1,
        Cpu = 2
    }
    // default props
    public int NumInferenceSteps = 15;
    public ExecutionProvider ExecutionProviderTarget = ExecutionProvider.DirectML;
    public double GuidanceScale = 7.5;
    public int Height = 512;
    public int Width = 512;
    public int DeviceId = 0;
    
    public string OrtExtensionsPath = "ortextensions.dll";
    public string TokenizerOnnxPath = "cliptokenizer.onnx";
    public string TextEncoderOnnxPath = "";
    public string UnetOnnxPath = "";
    public string VaeDecoderOnnxPath = "";
    public string SafetyModelPath = "";

    // default directory for images
    public string ImageOutputPath = "";

    public SessionOptions GetSessionOptionsForEp()
    {
        var sessionOptions = new SessionOptions();

        switch (this.ExecutionProviderTarget)
        {
            case ExecutionProvider.Cuda:
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                sessionOptions.AppendExecutionProvider_CUDA(this.DeviceId);
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            case ExecutionProvider.Cpu:
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
            case ExecutionProvider.DirectML:
            default:
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                sessionOptions.EnableMemoryPattern = false;
                sessionOptions.AppendExecutionProvider_DML(this.DeviceId);
                sessionOptions.AppendExecutionProvider_CPU();
                return sessionOptions;
        }
    }
}