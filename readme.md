# Hardware-Accelerated Transformer for Network Intrusion Detection on Edge Devices

## 🏆 Project Highlight

[cite_start]This research project, developed at the **I2CS Research Lab, Indian Institute of Information Technology, Kottayam**, has been officially selected for presentation at the prestigious **Design and Verification Conference (DVCon) India 2025**[cite: 1, 239]. This repository contains the complete software model, quantization analysis, and verification framework that serves as the high-level counterpart to our custom-designed hardware accelerator.

---

## 1. Introduction: The Problem

Transformer models have become the state-of-the-art in many machine learning domains, including Network Intrusion Detection Systems (NIDS). However, their effectiveness comes at a high computational cost. [cite_start]The core of a Transformer is the **Multi-Head Attention** mechanism, which relies heavily on large matrix multiplication operations (for Query, Key, and Value matrices) and subsequent feed-forward networks[cite: 135, 136].

[cite_start]This computational demand makes it extremely challenging to deploy these powerful models on **resource-constrained edge devices**, where power, memory, and processing capabilities are limited[cite: 140]. Our project directly addresses this challenge by offloading these expensive computations to a dedicated, high-performance hardware accelerator implemented on an FPGA.

---

## 2. Our Solution: A Hardware/Software Co-Design Approach

We have designed a complete end-to-end system that encompasses both the high-level software model and the low-level hardware implementation. This repository represents the **software half** of this co-design.

### The Hardware Accelerator

[cite_start]The core of our project is a custom **FPGA-based accelerator IP** designed to accelerate the matrix multiplication bottlenecks in Transformer models[cite: 138, 149]. Key features of our hardware design include:

* **Tiled Matrix Multiplication:** The accelerator uses a tiled computation approach, breaking large matrices into smaller 16x16 tiles. [cite_start]This strategy maximizes data reuse from fast on-chip memory (BRAMs) and minimizes slow, power-hungry access to external DDR3 memory[cite: 139, 157, 179].
* [cite_start]**Massive Parallelism:** The inner computation loops are fully unrolled using High-Level Synthesis (HLS) pragmas, creating **256 parallel Multiply-Accumulate (MCA) engines** that utilize the FPGA's DSP slices for high-throughput computation[cite: 159, 180, 433].
* [cite_start]**Pipelined Architecture:** The entire computation pipeline is optimized with an Initiation Interval (II) of 1, meaning it can accept new data every clock cycle, maximizing hardware utilization[cite: 159, 182].
* [cite_start]**Optimized Memory Access:** The design features three dedicated full AXI channels for concurrently reading input matrices and writing results, eliminating memory arbitration bottlenecks and enabling efficient AXI burst transfers[cite: 168, 189].

### The Software Framework (This Repository)

This repository provides the essential software components that enable and validate the hardware accelerator:

1.  **High-Level Model:** A functional **FlowTransformer model** built in TensorFlow/Keras, which serves as the golden reference for verifying the hardware's correctness.
2.  [cite_start]**Quantization Analysis:** The process of converting the model from high-precision 32-bit floating-point (FP32) numbers to hardware-friendly 8-bit integers (INT8)[cite: 14]. [cite_start]This is crucial for reducing the model's memory footprint and enabling efficient integer-based computation on the FPGA[cite: 164]. This repository contains the analysis for both **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)**.
3.  **Verification and Export:** The scripts generate test data, evaluate the performance of the quantized models, and export the final, optimized `INT8` model to the industry-standard **ONNX format** for deployment.

---

## 3. Project Workflow and Report Generation

This section provides a guide on how to use the notebooks to reproduce our analysis and a summary of the key observations from our reports.

### Notebook Guides

* **[`main_analysis.ipynb`](./main_analysis.ipynb):** This notebook is the starting point for understanding the baseline `FP32` model. It walks through defining the model's components, loading the dataset, and running a full training/evaluation cycle. Use this to understand the original model architecture and performance.
* **[`model_analysis.ipynb`](./model_analysis.ipynb):** Use this notebook to load a previously saved `.keras` model and evaluate its performance on the test set. It provides a detailed breakdown of accuracy, a classification report, and a confusion matrix.
* **[`quantization_analysis.ipynb`](./quantization_analysis.ipynb):** This is the main workflow for this project. Run this notebook cell-by-cell to perform the complete quantization analysis, which includes establishing the FP32 baseline, performing QAT, fine-tuning, and exporting the final `INT8` ONNX model.

### Key Observations and Results

The project successfully validated the hardware/software co-design methodology. The software analysis, documented in the `quantization_analysis.ipynb` notebook, yielded the following key results:

* **Baseline Model Performance:** The initial FP32 model struggled with the imbalanced dataset, achieving a very low **F1-score of 0.10** for the critical "Malicious" class. This highlighted the need for further optimization.

* **Quantization-Aware Training (QAT) Success:** After converting the model to an INT8 representation using QAT and fine-tuning for a single epoch, the performance improved dramatically:
    * **Final F1-Score:** The QAT model achieved an **F1-score of 0.96**, indicating a highly accurate and reliable model for detecting threats.
    * **Performance Gain:** This represents a massive **F1-score gain of +0.86** over the baseline, demonstrating that the quantization process not only optimized the model but significantly improved its effectiveness.
    * **Model Size Reduction:** The final `INT8` model is only **3.03 MB**, a nearly **4x size reduction** from the ~12 MB FP32 model, making it ideal for deployment on resource-constrained edge devices.

* **Post-Training Quantization (PTQ):** This simpler quantization method was found to be unstable with the model's complex multi-input architecture, causing the TFLite converter to hang. The decision was made to proceed with the more robust and higher-performing QAT approach.

* **Hardware Synthesis:** As detailed in our DVCon reports, the accelerator IP was synthesized efficiently, utilizing only **31% of available DSPs** and **36% of Block RAM** for a large matrix multiplication test case, confirming its viability for FPGA implementation.
* **Hardware Synthesis:** The accelerator IP was successfully synthesized using Vitis HLS. [cite_start]For a test case with matrix sizes of 128x512 and 512x512, the design utilized only **31% of available DSP blocks** and **36% of Block RAM Tiles**, staying well within the resource constraints of the target edge device[cite: 393, 431, 443]. This demonstrates an efficient hardware implementation with room for future scalability.
* [cite_start]**Hardware Simulation:** RTL cosimulation in Vivado XSIM verified the functional correctness of the accelerator[cite: 202]. [cite_start]The results showed significant performance gains, with a computation time of approximately **47ms for a 128x512x512 matrix multiplication**, achieving **1.42 GFLOPs**[cite: 221]. [cite_start]The tiled writeback approach also showed improved AXI bus utilization by creating periodic gaps for other system components to access memory[cite: 85, 86].
* **Software Quantization:** The `FP32` software model was successfully quantized to an `INT8` representation using **Quantization-Aware Training (QAT)**. After fine-tuning for just one epoch, the QAT model showed a dramatic improvement in the F1-score for the minority "Malicious" class, indicating that the process not only reduced model size but also improved its detection capabilities. Post-Training Quantization (PTQ) was attempted but found to be unstable with this model's complex architecture.
* **Final Output:** The process culminates in the creation of a final, quantized `INT8` model in the **ONNX format (`final_int8_model.onnx`)**, ready for deployment and hardware verification.

---

## 4. Repository Structure & Key Files

* `framework/` & `implementations/`: Contains the core Python source code for the FlowTransformer model, including its custom layers like `TransformerEncoderBlock`.
* `saved_models/`: This directory contains the pre-trained `FP32` model (`.keras`), its configuration (`.json`), and the final quantized `INT8` model (`final_int8_model.onnx`).
* `quantization_analysis.ipynb`: The primary Jupyter Notebook containing the complete workflow for model analysis, quantization, and ONNX export.
* [cite_start]`*.pdf` & `*.docx`: These are the official **DVCon India 2025 reports** containing in-depth details about the hardware synthesis, resource utilization, and simulation results with waveforms [cite: 1-104, 105-238, 239-449].

---

## 5. Project Status & Next Steps

* ✅ **FPGA Accelerator Synthesis:** Complete.
* ▶️ **FPGA Accelerator Implementation:** In progress on the VEGA Processor.
* ✅ **Software Model Quantization:** Complete.

This repository is a critical component of our hardware/software co-design approach, enabling rapid analysis and verification of our accelerator design.

---

## 6. References

[cite_start][1] R. Li and S. Chen, "Design and Implementation of an FPGA-Based Hardware Accelerator for Transformer," arXiv preprint arXiv:2503.16731, 2025[cite: 232].

[2] X. Yang and T. Su, "EFA-Trans: An Efficient and Flexible Acceleration Architecture for Transformers," Electronics, vol. 11, no. 21, p. [cite_start]3550, 2022[cite: 233].

[3] Xilinx, "Matrix Multiplication | High Level Systhesis Design Flow," GitHub. [Online]. [cite_start]Available: https://github.com/Xilinx/xup_high_level_synthesis_design_flow/blob/main/source/matmult/notebook/matmul_part2.ipynb[cite: 235].

[4] Xilinx, "Vitis HLS User Guide (UG1399)," 2022. [Online]. [cite_start]Available: https://www.xilinx.com/support/documents/sw_manuals/xilinx2022_2/ug1399-vitis-hls.pdf[cite: 237].
# FlowTransformer
The framework for transformer based NIDS development

## Jupyter Notebook

We have included an example of using FlowTransformer with a fresh dataset in the Jupyter notebook available in [demonstration.ipynb](demonstration.ipynb)

## Usage instructions

FlowTransformer is a modular pipeline that consists of four key components. These components can be swapped as required for custom implementations, or you can use our supplied implementations:

| **Pre-Processing** | **Input Encoding** | **Model** | **Classification Head** |
|--------------------|--------------------|-----------|-------------------------|
| The pre-processing component accepts arbitrary tabular datasets, and can standardise and transform these into a format applicable for use with machine learning models. For most datasets, our supplied `StandardPreprocessing` approach will handle datasets with categorical and numerical fields, however, custom implementations can be created by overriding `BasePreprocessing`                  | The input encoding component will accept a pre-processed dataset and perform the transformations neccescary to ingest this as part of a sequence to sequence model. For example, the embedding of fields into feature vectors.                  | FlowTransformer supports the use of any sequence-to-sequence machine learning model, and we supply several Transformer implementations.         | The classification head is responsible for taking the sequential output from the model, and transforming this into a fixed length vector suitable for use in classification. We recommed using `LastToken` for most applications.                       |

To initialise FlowTransformer, we simply need to provide each of these components to the FlowTransformer class:
```python
ft = FlowTransformer(
  pre_processing=...,
  input_encoding=...,
  sequential_model=...,
  classification_head=...,
  params=FlowTransformerParameters(window_size=..., mlp_layer_sizes=[...], mlp_dropout=...)
)
```

The FlowTransformerParameters allows control over the sequential pipeline itself. `window_size` is the number of items to ingest in a sequence, `mlp_layer_sizes` is the number of nodes in each layer of the output MLP used for classification at the end of the pipeline, and the `mlp_dropout` is the dropout rate to apply to this network (0 for no dropout). 

FlowTransformer can then be attached to a dataset, doing this will perform pre-processing on the dataset if it has not already been applied (caching is automatic):

```python
ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)
```

Once the dataset is loaded, and the input sizes are computed, a Keras model can be built, which consists of the `InputEncoding`, `Model` and `ClassificationHead` components. To do  this, simply call `build_model` which returns a `Keras.Model`:

```python
model = ft.build_model()
model.summary()
```

Finally, FlowTransformer has a built in training and evaluation method, which returns pandas dataframes for the training and evaluation results, as well as the final epoch if early stopping is configured:

```python
(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=5, steps_per_epoch=64, early_stopping_patience=5)
```

However, the `model` object can be used in part of custom training loops. 

## Implementing your own solutions with FlowTransformer

### Ingesting custom data formats

Custom data formats can be easily ingested by FlowTransformer. To ingest a new data format, a `DataSpecification` can be defined, and then supplied to `FlowTransformer`:

```python
dataset_spec = DatasetSpecification(
    include_fields=['OUT_PKTS', 'OUT_BYTES', ..., 'IN_BYTES', 'L7_PROTO'],
    categorical_fields=['CLIENT_TCP_FLAGS', 'L4_SRC_PORT', ..., 'L4_DST_PORT', 'L7_PROTO'],
    class_column="Attack",
    benign_label="Benign"
)

flow_transformer.load_dataset(dataset_name, path_to_dataset, dataset_spec) 
```

The rest of the pipeline will automatically handle any changes in data format - and will correctly differentiate between categorical and numerical fields.

### Implementing Custom Pre-processing 

To define a custom pre-processing (which is generally not required, given the supplied pre-processing is capable of handling the majority of muiltivariate datasets), override the base class `BasePreprocessing`:

```python
class CustomPreprocessing(BasePreProcessing):

    def fit_numerical(self, column_name:str, values:np.array):
        ...

    def transform_numerical(self, column_name:str, values: np.array):
        ...

    def fit_categorical(self, column_name:str, values:np.array):
        ...

    def transform_categorical(self, column_name:str, values:np.array, expected_categorical_format:CategoricalFormat):
        ...
```

Note, the `CategoricalFormat` here is passed automatically by the `InputEncoding` stage of the pipeline:
- If the `InputEncoding` stage expects categorical fields to be encoded as integers, it will return `CategoricalFormat.Integers`
- If the `InputEncoding` stage expets categorical fields to be one-hot encoded, it will return `CategoricalFormat.OneHot`

Both of these cases must be handled by your custom pre-processing implementation.

### Implementing Custom Encodings 

To implement a custom input encoding, the `BaseInputEncoding` class must be overridden. 

```python
class CustomInputEncoding(BaseInputEncoding):
    def apply(self, X:List["keras.Input"], prefix: str = None):
        # do operations on the inputs X
        ...
        return X

    @property
    def required_input_format(self) -> CategoricalFormat:
        return CategoricalFormat.Integers
```

Here, `apply` is simply the input encoding tranformation to be applied to the inputs to the model. For no transformation, we can simply return the input. The required input format should return the expected format of categorical fields, if this should be `Integers` or `OneHot`.

### Implementing Custom Transformers

Custom transformers, or any sequential form of machine learning model can be implemented by overriding the `BaseSequential` class:
```python
class CustomTransformer(BaseSequential):
    
    @property
    def name(self) -> str:
        return "TransformerName"
        
    @property
    def parameters(self) -> dict:
        return {
            # ... custom parameters ... eg:
            # "n_layers": self.n_layers,
        }
       
    def apply(self, X, prefix: str = None):
        m_X = X
        # ... model operations on X ...
        return m_X     
```

### Implementing Custom Classification Heads

To implement a custom classification head, override the BaseClassificationHead class. Here two methods can be overriden:

```python
class CustomClassificationHead(BaseClassificationHead):
    def apply_before_transformer(self, X, prefix:str=None):
        # if any processing must be applied to X before being passed to 
        # the transformer, it can be done here. For example, modifying
        # the token format to include additional information used by the
        # classification head.
        return X
    
    
    def apply(self, X, prefix: str = None):
        # extract the required data from X
        return X
```

## Currently Supported FlowTransformer Components

Please see the wiki for this Github for a list of the associated FlowTransformer components and their description. Feel free to expand the Wiki with your own custom components after your pull request is accepted.

## Datasets used in this work

Several of the datasets used in this work [are available here](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
