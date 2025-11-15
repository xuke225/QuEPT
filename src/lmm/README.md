## Installation
1. Install packages and 3rdparty repos
    ```bash
    # Install LLaVA-NeXT
    mkdir 3rdparty
    cd 3rdparty
    git clone https://github.com/LSY-noya/LLaVA-NeXT.git
    cd LLaVA-NeXT
    pip install -e .
    cd ..

    # Install lmms-eval
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    cd lmms-eval
    pip install -e .
    ```
    Ref:  
    * https://github.com/LLaVA-VL/LLaVA-NeXT
    * https://github.com/EvolvingLMMs-Lab/lmms-eval
2. Dataset
   We follow the settings of MBQ: Modality-Balanced Quantization for Large Vision-Language Models [CVPR2024]

   Cailb Download Link: https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md#prepare-images 
   
   Detailed settings please ref: https://github.com/thu-nics/MBQ
   
## Run Quantization
    ```bash
    bash run_llava_wa.sh
    ```