cd src/open-r1-multimodal 
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation
pip install transformers==4.51.3

# RALI
pip install joblib
pip install scikit-learn