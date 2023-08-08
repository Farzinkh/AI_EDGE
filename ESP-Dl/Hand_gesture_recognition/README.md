# Deploy a Model Step by Step

This tutorial shows how to deploy a model with the [my Toolkit](../Tools/model_builder.py) and run benchmark using SD Card.

Note: For a model quantized on other platforms:
- If the quantization scheme (e.g. TFLite int8 model) is different from that of ESP-DL then the model cannot be deployed with ESP-DL.


# Preparation

# step 1: Train [Hand Gesture Recognition](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) model

<p align="center">
    <img width="%" src="./logs/handrecognition_model_int8.onnx.png"> 
</p>


An example of how to train and convert your model to ONNX is [here](https://colab.research.google.com/drive/1v0z9TtTr9b73pfd51tNwvzFB0xPg1D6F?usp=sharing).


## Step 1.1: Convert Your Model

In order to be deployed, the trained floating-point model must be converted to an integer model, the format compatible with ESP-DL. Given that ESP-DL uses a different quantization scheme and element arrangements compared with other platforms, 

## Step 1.2: Config project by [build_scripts_config.toml](./build_scripts_config.toml)

To faciliate next steps a config file is provided which is required by model_builder script.

For example chnage "Format" to "int16" if you trained model by int16 dataset or change "SoC" to "ESP32" if you want to deploy model on ESP32.

## Step 1.3: Convert to ESP-DL Format and Quantize

The calibrator in the quantization toolkit can quantize a floating-point model to an integer model which is compatible with ESP-DL. For post-training quantization, please prepare the calibration dataset (can be the subset of training dataset or validation dataset).

Run the script with the following command in Tools directory:

```python
python model_builder.py -p <Hand Gesture Recognition project directory>
```

And you will see the following log which includes the quantized coefficients for the model's input and output. These coefficients will be used in later steps when defining the model so enter them in order from the begining layer until the last layer seprating by ',' for example:

```python

Quantized model info:
model input name: conv2d_input, exponent: 1
Reshape layer name: StatefulPartitionedCall/sequential/conv2d/BiasAdd__6, output_exponent: 1
Conv layer name: StatefulPartitionedCall/sequential/conv2d/BiasAdd, output_exponent: 1
MaxPool layer name: StatefulPartitionedCall/sequential/max_pooling2d/MaxPool, output_exponent: 1
Transpose layer name: StatefulPartitionedCall/sequential/max_pooling2d/MaxPool__12, output_exponent: 1
Reshape layer name: StatefulPartitionedCall/sequential/flatten/Reshape, output_exponent: 1
Gemm layer name: fused_gemm_0, output_exponent: 1
Gemm layer name: fused_gemm_1, output_exponent: 0
Softmax layer name: StatefulPartitionedCall/sequential/dense_1/Softmax, output_exponent: -6

```

Now you will asked to enter `1,1,1,0,-6` as desired exponents. note that you dont need to enter reshape,transpose and maxpool layers exponent values.

For more information about quantization toolkit API, please refer to [Quantization Toolkit API](https://github.com/espressif/esp-dl/blob/master/tools/quantization_tool/quantization_tool_api.md).

# Step 2: Compile & Deploy project

`idf.py set-target esp32s3` or `idf.py set-target esp32`

`idf.py menuconfig` => for more configuration (optional).

`idf.py build flash monitor`


# Step 3: Run Your Model

In terminal after ESP booted enter `run_benchmark` command or enter `help` to receive more information. After the benchmark is finished, all the records are automatically copied to the sdcard as `report.txt` file so that you can extract it on a PC.


# Results
Finally, to evaluate result run [analysis script](../Tools/analysis.py) and provide actual path of record file.
```python
 python analysis.py -l <Path to REPORT.TXT>
```

<center>

| SoC      | description         | delay  | accuracy |
|----------|:-------------------:|:------:|---------:|
| ESP32    | INT8 | 527ms   | 99.83    |
| ESP32S3  | INT8 | 403ms   | 99.75

</center>

</center>

<p align="center">
    <img width="%" src="./logs/Quantization.png">
</p>
<p align="center">
    <img width="%" src="./logs/REPORT_1200_INT8.png"> 
</p>