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

## Example queries
When the solution runs, it will start an ASP.NET webservice at localhost:5001.

|    | Context | Question | Model Reply |
| -- | --  | -- | -- |
| ([link](https://localhost:5001/api/predict?Context=Bob%20is%20walking%20through%20the%20woods%20collecting%20blueberries%20and%20strawberries%20to%20make%20a%20pie.&Question=What%20is%20his%20name?)) | Bob is walking through the woods collecting blueberries and strawberries to make a pie. | What is his name? | ✅ `{"tokens":["bob"],"probability":0.8884454}` |
| ([link](https://localhost:5001/api/predict?Context=Bob%20is%20walking%20through%20the%20woods%20collecting%20blueberries%20and%20strawberries%20to%20make%20a%20pie.&Question=What%20will%20he%20bring%20home?)) | Bob is walking through the woods collecting blueberries and strawberries to make a pie. | What will he bring home? | ✅ `{"tokens":["blueberries","and","strawberries"],"probability":0.4070111}` |
| ([link](https://localhost:5001/api/predict?Context=Bob%20is%20walking%20through%20the%20woods%20collecting%20blueberries%20and%20strawberries%20to%20make%20a%20pie.&Question=Where%20is%20bob?)) | Bob is walking through the woods collecting blueberries and strawberries to make a pie. | Where is Bob? | ✅ `{"tokens":["walking","through","the","woods"],"probability":0.6123137}` |
| ([link](https://localhost:5001/api/predict?Context=Bob%20is%20walking%20through%20the%20woods%20collecting%20blueberries%20and%20strawberries%20to%20make%20a%20pie.&Question=What%20will%20he%20bake?)) | Bob is walking through the woods collecting blueberries and strawberries to make a pie. | What will he bake? | ❌ `{"tokens":["blueberries","and","strawberries"],"probability":0.48385787}` |



