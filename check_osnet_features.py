import os
import torch
import torchreid

def test_osnet_features():
    print("=" * 60)
    print("STEP 4: OSNet x1.0 Pretrained Weights & Feature Map Extraction")
    print("=" * 60)
    
    # 1. Build osnet_x1_0 with pretrained=True
    print("Building torchreid osnet_x1_0 (pretrained=True)...")
    try:
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
    except Exception as e:
        print(f"Error building model with default downloader: {e}")
        print("Attempting robust download fallback...")
        # TorchReID uses gdown to download from Google Drive
        # URL: https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY
        import gdown
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.cache/torch'))
        ckpt_dir = os.path.join(torch_home, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        target_ckpt = os.path.join(ckpt_dir, 'osnet_x1_0_imagenet.pth')
        
        if not os.path.exists(target_ckpt):
            gdrive_url = 'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY'
            gdown.download(gdrive_url, target_ckpt, quiet=False)
            
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        
    model.eval()
    print("OSNet x1.0 built and loaded successfully!")
    
    # 2. Test feature map extraction for a random (2, 3, 224, 224) input tensor
    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"\nInput tensor shape: {dummy_input.shape}")
    
    with torch.no_grad():
        # featuremaps extracts the tensor before global pooling
        feat_maps = model.featuremaps(dummy_input)
        
    print(f"Extracted feature map shape (before global pooling): {feat_maps.shape}")
    b, c, h, w = feat_maps.shape
    print(f" - Batch size (B): {b}")
    print(f" - Feature Channels (C): {c}")
    print(f" - Spatial Height (H): {h}")
    print(f" - Spatial Width (W): {w}")
    print(f"\nSpatial size available for 3x3 grid pooling: {h} x {w} (Channels: {c})")
    if h >= 3 and w >= 3:
        print(f"[OK] Spatial dimension {h}x{w} is sufficient for 3x3 grid pooling!")
    else:
        print(f"[WARNING] Spatial dimension {h}x{w} may be too small for 3x3 grid pooling without adaptive pooling.")

if __name__ == "__main__":
    test_osnet_features()
