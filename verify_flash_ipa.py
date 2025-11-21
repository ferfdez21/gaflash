
import torch
from gafl.models.flash_ipa.model import Model, ModelConfig
from gafl.models.flash_ipa.ipa import IPAConfig
from gafl.models.flash_ipa.edge_embedder import EdgeEmbedderConfig

def verify_flash_ipa():
    print("Verifying Flash IPA Model...")
    
    # Create dummy config
    ipa_conf = IPAConfig(
        use_flash_attn=True,
        c_s=256,
        c_z=128,
        c_hidden=128,
        no_heads=4,
        z_factor_rank=2,
        num_blocks=1
    )
    edge_conf = EdgeEmbedderConfig(
        c_s=256,
        c_p=128,
        feat_dim=64,
        mode="flash_1d_bias",
        z_factor_rank=2,
        k=10,
        num_bins=22
    )
    
    model_conf = ModelConfig(
        mode="flash_1d_bias",
        node_embed_size=256,
        edge_embed_size=128,
        ipa=ipa_conf,
        edge_features=edge_conf
    )
    
    # Instantiate model
    try:
        model = Model(model_conf)
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return

    # Create dummy input
    B, L = 1, 100
    input_feats = {
        'res_mask': torch.ones((B, L)),
        't': torch.rand((B,)),
        'trans_t': torch.rand((B, L, 3)),
        'rotmats_t': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1),
        'res_idx': torch.arange(L).unsqueeze(0).repeat(B, 1)
    }
    
    # Run forward pass
    try:
        print("Running forward pass...")
        # Move to GPU if available for flash attn
        if torch.cuda.is_available():
            model = model.cuda()
            for k, v in input_feats.items():
                input_feats[k] = v.cuda()
            # Flash attn requires fp16 or bf16 usually
            model = model.bfloat16()
            for k, v in input_feats.items():
                if v.dtype == torch.float32:
                    input_feats[k] = v.bfloat16()
                    
        out = model(input_feats)
        print("Forward pass successful.")
        print("Output keys:", out.keys())
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_flash_ipa()
