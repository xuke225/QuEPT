# QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching

Elastic precision quantization enables multi-bit deployment via a single optimization pass, fitting diverse quantization scenarios. Yet, the high storage and optimization costs associated with the Transformer architecture, research on elastic quantization remains limited, particularly for large language models. This paper proposes QuEPT, an efficient posttraining scheme that reconstructs block-wise multi-bit errors with one-shot calibration on a small data slice. It can dynamically adapt to various predefined bit-widths by cascading different low-rank adapters, and supports real-time switching between uniform quantization and mixed precision quantization without repeated optimization. To enhance accuracy and robustness, we introduce Multi-Bit Token Merging (MB-ToMe) to dynamically fuse token features across different bit-widths, improving robustness during bit-width switching. Additionally, we propose Multi-Bit Cascaded Low-Rank adapters (MB-CLoRA) to strengthen correlations between bit-width groups, further improve the overall performance of QuEPT. Extensive experiments demonstrate that QuEPT achieves comparable or better performance to existing state of-the-art post-training quantization methods.


<h1 align="center">   
    <img src="./img/quept.png" width="1000">  
</h1>  

## Runing Example
* For Vision Transformers (ViTs) quantization:
    ```shell
    cd src/vit/
    pip install -r requirements.txt
    bash test_quant.sh
    ```

* For Large Language Models (LLMs) quantization:
    ```shell
    cd src/llm/
    pip install -r requirements.txt
    git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
    cd fast-hadamard-transform
    pip install -e .
    cd ..
    bash run_weight_act.sh
    ```

* For Large Multimodal Models (LMMs) / Vision Language Models (VLMs):
    * Install packages and 3rdparty repos
    ```bash
    # Install LLaVA-NeXT
    cd src/lmm/
    mkdir 3rdparty
    git clone https://github.com/LSY-noya/LLaVA-NeXT.git
    cd LLaVA-NeXT
    pip install -e .
    cd ..

    # Install lmms-eval
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    cd lmms-eval
    pip install -e .

    pip install -r requirements.txt

    bash run_llava_wa.sh
    ```
    Ref:  
    * https://github.com/LLaVA-VL/LLaVA-NeXT
    * https://github.com/EvolvingLMMs-Lab/lmms-eval
  
    * Dataset
       * We follow the settings of MBQ: Modality-Balanced Quantization for Large Vision-Language Models

       * Cailb Download Link: https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md#prepare-images 
        
       * Detailed settings please ref: https://github.com/thu-nics/MBQ

## Citation
```
@inproceedings{
    quept,
    title={QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching},
    author={Xu, Ke and Wang, Yixin and Li, Zhongcheng and Cui, Hao and Hu, Jinshui and Zhang, Xingyi},
    booktitle={AAAI},
    year={2026}
}
```