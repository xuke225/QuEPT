import timm

def build_model(name):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.
    Currently support almost all models in timm.models.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    - deit_base(_distilled)_patch16_384,
    - swin_tiny/small/base/large_patch4_window7_224,
    - swin_base/large_patch4_window12_384
    These models are finetuned on imagenet-1k and should use ViTImageNetLoaderGenerator
    for calibration and testing.
    """
    model = timm.create_model(name, pretrained=True)

    return model
