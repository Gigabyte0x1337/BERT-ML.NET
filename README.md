# BERT-ML.NET
Question and Answering (Q&A) BERT model implimentation for ML.NET.

An example of BERT model predictions in .NET Core/.NET Standard.

## Model
https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad

Download the pre-trained BERT ONNX model by running `getDependicies.sh`. Or download the [model](https://github.com/onnx/models/blob/master/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx?raw=true) to the `BERT.WebApi/Model` folder.

## GPU usage
The project is setup to run on CPU. This allows the sample to run on machines without an Nvidia GPU.

To run on an Nvidia CUDA GPU:
* Set `hasGpu = true` in OnnxModelConfigurator.cs
* Remove NuGet `Microsoft.ML.OnnxRuntime.NoOpenMP`
* Add NuGet `Microsoft.ML.OnnxRuntime.Gpu`
