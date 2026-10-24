import os
import sys
import socket
import torch
import torchreid

def block_network():
    def guard(*args, **kwargs):
        raise RuntimeError("Network call blocked during OFFLINE test!")
    socket.socket = guard

def main():
    print("=" * 60)
    print("STEP 7: OFFLINE VERIFICATION CHECK FOR OSNET")
    print("=" * 60)
    
    ckpt_path = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth")
    if not os.path.exists(ckpt_path):
        print(f"[FAIL] Checkpoint not found at: {ckpt_path}")
        sys.exit(1)
        
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"[OK] Found cached weights: {ckpt_path} ({size_mb:.2f} MB)")
    
    print("\nEnforcing network block...")
    block_network()
    print("Network sockets blocked.")
    
    print("\nLoading osnet_x1_0(pretrained=True) offline...")
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        pretrained=True
    )
    model.eval()
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model.featuremaps(x)
        
    print(f"Extracted feature map shape: {out.shape}")
    print("\n" + "=" * 60)
    print("OFFLINE CHECK PASSED: OSNet loaded and ran offline with ZERO network calls!")
    print("=" * 60)

if __name__ == "__main__":
    main()
