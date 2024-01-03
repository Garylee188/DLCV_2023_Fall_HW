#!/bin/bash

# p1
wget -O diffusion_model.pth 'https://www.dropbox.com/scl/fi/katqtx49yrbxt6cm0ncul/model_e20.pth?rlkey=0ol2mide0gz3zlekmkv6skciu&dl=1'

# p3 svhn
wget -O svhn_feature_extractor.ckpt 'https://www.dropbox.com/scl/fi/dl6kdpblvor0smzlau1et/SVHN_20_Feature_Extractor.ckpt?rlkey=ugroye7gy7x9ehqfe28mtrbwe&dl=1'
wget -O svhn_label_predictor.ckpt 'https://www.dropbox.com/scl/fi/bydyjzs11t59nj1jp0d53/SVHN_20_Label_Predictor.ckpt?rlkey=i3x0sgfbu2kuybywh6elx3en0&dl=1'
wget -O svhn_domain_classifier.ckpt 'https://www.dropbox.com/scl/fi/fk5dot2co1qr2174kn7tn/SVHN_20_Domain_Classifier.ckpt?rlkey=v0wbb1agb7nzcbfe77pviwj5m&dl=1'

# p3 usps
wget -O usps_feature_extractor.ckpt 'https://www.dropbox.com/scl/fi/sx6fs5hwqsn0ll8n9zz52/USPS_20_Feature_Extractor.ckpt?rlkey=o1l4rzebhlvfiejjjkmginivo&dl=1'
wget -O usps_label_predictor.ckpt 'https://www.dropbox.com/scl/fi/cf80xpie4lfixth4bl81e/USPS_20_Label_Predictor.ckpt?rlkey=crx4xovg8uvxhvgdhymtkn9z2&dl=1'
wget -O usps_domain_classifier.ckpt 'https://www.dropbox.com/scl/fi/hr1j3tmqity3haim33rnv/USPS_20_Domain_Classifier.ckpt?rlkey=0hxz95vcp1aj7ces3ou55mpdu&dl=1'