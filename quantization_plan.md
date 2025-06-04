# TODO: ConditionalUNet Quantization Plan

This document outlines the steps to implement quantization for the `ConditionalUNet` model, aiming for optimal performance (inference speed, model size) while minimizing any degradation in generated image quality.

## 1. Establish Baseline Performance (FP32 Model)

*   [ ] **Define Evaluation Metrics:**
    *   Image quality metrics: FID (Fréchet Inception Distance), LPIPS (Learned Perceptual Image Patch Similarity), PSNR, SSIM.
    *   Visual inspection: Subjective assessment of generated image quality.
    *   Performance metrics: Inference latency (ms/image), throughput (images/sec), model size (MB).
*   [ ] **Benchmark FP32 Model:**
    *   Run inference on a representative validation dataset.
    *   Collect and record all defined metrics for the current FP32 model. This will be our baseline.
*   [ ] **Profile FP32 Model:**
    *   Use PyTorch Profiler to identify bottlenecks (CPU/GPU time per layer, memory usage). This can help target which parts of the model might benefit most or be most sensitive to quantization.

## 2. Explore Mixed Precision (FP16/BFloat16)

This is often a good first step for performance improvement with minimal effort and usually negligible quality loss. BFloat16 is generally preferred on newer hardware that supports it.

*   [ ] **Implement Automatic Mixed Precision (AMP):**
    *   Integrate `torch.cuda.amp.GradScaler` (if training) and `torch.cuda.amp.autocast` (for training and inference) into your training and inference pipelines.
*   [ ] **Evaluate Mixed Precision Model:**
    *   Re-run benchmarks and collect all defined metrics.
    *   Compare against the FP32 baseline.
    *   Assess any changes in image quality.
*   [ ] **Decision Point:** If FP16/BFloat16 provides sufficient speedup/memory reduction with acceptable quality, this might be a good stopping point or a new baseline for further INT8 quantization.

## 3. Prepare for INT8 Quantization

*   [ ] **Study PyTorch Quantization Toolkit:**
    *   Familiarize with `torch.quantization` API, including:
        *   `torch.quantization.prepare_qat` (for QAT)
        *   `torch.quantization.prepare` (for PTQ static)
        *   `torch.quantization.convert`
        *   `torch.quantization.fuse_modules`
        *   QuantStub, DeQuantStub
        *   Quantizable versions of layers (e.g., `torch.ao.nn.quantized.Linear`, `torch.ao.nn.quantized.Conv2d`).
*   [ ] **Model Analysis and Modification:**
    *   Review `ConditionalUNet`, `ResidualBlock`, `AttentionBlock`, and `SinusoidalPositionalEncoding`.
    *   Identify layers that can be quantized (Conv2d, Linear).
    *   Ensure custom modules or operations not directly supported by quantization are handled:
        *   They might need to be rewritten to use quantizable components.
        *   Alternatively, parts of the model can be left in FP32 (partial quantization) by inserting `QuantStub` and `DeQuantStub` appropriately.
    *   `SinusoidalPositionalEncoding`: Analyze if this can be quantized or should remain FP32. Lookup tables can sometimes be quantized effectively.
    *   `nn.GroupNorm`: Check current support status in PyTorch quantization. May need to be replaced or handled carefully (e.g., kept in FP32 or use `FusedBatchNorm` if applicable before quantization).
    *   `nn.SiLU`: Ensure it's handled correctly by the quantization process (often fused).
*   [ ] **Implement Layer Fusion:**
    *   Identify patterns like Conv-BatchNorm-ReLU (or Conv-SiLU) and apply `torch.quantization.fuse_modules` to improve quantization accuracy and performance. This should be done *before* preparing the model for quantization.

## 4. Post-Training Static Quantization (PTQ) - INT8

PTQ is simpler to implement than QAT as it doesn't require retraining, but might lead to more accuracy drop.

*   [ ] **Prepare Model for PTQ:**
    *   Insert `QuantStub` at the input and `DeQuantStub` at the output of the model (or sections to be quantized).
    *   Replace float modules with their `torch.ao.nn.quantizable` counterparts if necessary or ensure the model structure is compatible.
    *   Specify a quantization configuration (qconfig), e.g., `torch.quantization.get_default_qconfig('fbgemm')` for x86 or `'qnnpack'` for ARM.
    *   Call `torch.quantization.prepare(model_ptq, inplace=True)`.
*   [ ] **Calibration:**
    *   Feed a representative subset of your training or validation data (unlabeled is fine) through the prepared model. This allows PyTorch to observe activation ranges and determine optimal quantization parameters (scale and zero-point).
*   [ ] **Convert to Quantized Model:**
    *   Call `torch.quantization.convert(model_ptq, inplace=True)`.
*   [ ] **Evaluate PTQ INT8 Model:**
    *   Run benchmarks, collect metrics, and compare against the FP32/FP16 baseline.
    *   Thoroughly assess image quality.

## 5. Quantization-Aware Training (QAT) - INT8

QAT typically yields better results for INT8 quantization as it simulates quantization effects during training, allowing the model to adapt.

*   [ ] **Prepare Model for QAT:**
    *   Similar to PTQ, insert `QuantStub`/`DeQuantStub`.
    *   Specify a QAT configuration, e.g., `torch.quantization.get_default_qat_qconfig('fbgemm')`.
    *   Call `torch.quantization.prepare_qat(model_qat, inplace=True)`.
*   [ ] **Fine-tuning/Training:**
    *   Fine-tune the QAT-prepared model on your training dataset for a number of epochs. Start with a pre-trained FP32 model.
    *   Alternatively, train from scratch with QAT enabled (less common if a good FP32 model exists).
*   [ ] **Convert to Quantized Model:**
    *   After training, switch the model to evaluation mode (`model_qat.eval()`).
    *   Call `torch.quantization.convert(model_qat, inplace=True)`.
*   [ ] **Evaluate QAT INT8 Model:**
    *   Run benchmarks, collect metrics, and compare. This is expected to be the best INT8 result.

## 6. Deployment and Backend Considerations

*   [ ] **Save/Load Quantized Models:**
    *   Understand how to properly save and load quantized model states using `torch.save` and `torch.load_state_dict`.
*   [ ] **Target Backend:**
    *   Ensure the chosen quantization configuration (`fbgemm`, `qnnpack`) matches the target deployment hardware for optimal performance.
*   [ ] **Consider ONNX Export (Optional):**
    *   If deploying to environments that use ONNX Runtime, explore exporting the quantized model to ONNX format. PyTorch has built-in support for this.

## 7. Iteration and Debugging

*   [ ] **Iterate:** If quality degradation is too high with full INT8 quantization, consider:
    *   Partial quantization (keeping sensitive layers in FP16/FP32).
    *   Revisiting QAT fine-tuning strategy (learning rate, number of epochs).
    *   Trying different quantization configurations or observers.
*   [ ] **Debugging:** Use tools and techniques to debug quantization issues, such as visualizing weights and activation distributions.

This plan provides a structured approach. Depending on your initial results with mixed precision or PTQ, you might decide not to proceed with all steps if satisfactory performance/quality is achieved earlier. 