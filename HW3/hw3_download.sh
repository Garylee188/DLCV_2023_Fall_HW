#!/bin/bash

# load pre-trained model
python3 -c "import clip; clip.load('ViT-B/32')"
python3 -c "import timm; timm.create_model('vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k', pretrained=True, num_classes=0)"

# p2 model
wget -O best_vl_model_huge_lora.pth 'https://www.dropbox.com/scl/fi/gq010dhnsg70hkdaybsiv/best_vl_model_huge_lora.pth?rlkey=ccvb8mw7gbm4rgmfswsr3lyo0&dl=1'
