# AIDA - Multimodal Event Coreference - Multiple Instace Learning with Objects

### Environment Setup for DVMM Lab Members
``` 
source /dvmm-filer2/users/shoya/anaconda3/bin/activate
conda activate py37
``` 

# Multimodal Event Coreference 

- All files for training the models are under /home/shoya/howto100m

## Multimodal Event Coreference WITHOUT proposals
m2e2_train_milo_eval_coref_with_no_proposals.py
 ``` 
CUDA_VISIBLE_DEVICES=6 python m2e2_train_milo_eval_coref_with_no_proposals.py \
--m2e2=1 --eval_m2e2=1 --num_thread_reader=16 --batch_size=1 --epochs=20 --n_display=400 --lr_decay=1.0 --embd_dim=6144 \
--pretrain_path=model/howto100m_pt_model.pth \
--videoID2obj_feature_path /dvmm-filer2/projects/AIDA/data/m2e2_video_features/metadata/obj_det_outputs/additional_vids/ten_feats_per_frame/additional_vids2all_ASRs.json \
--video_clip_feature_dir /dvmm-filer2/projects/AIDA/data/m2e2_video_features/event_coreference_clips/video_feats \
--m2e2_manual_labels_obj_features_json_val  /dvmm-filer2/projects/AIDA/data/m2e2_video_features/metadata/obj_det_outputs/event_coreference_clips/m2e2_obj_detections_event_coreference_with_all_article_sentences_event_coref_propagated_val.json \
--m2e2_manual_labels_obj_features_json_test  /dvmm-filer2/projects/AIDA/data/m2e2_video_features/metadata/obj_det_outputs/event_coreference_clips/m2e2_obj_detections_event_coreference_with_all_article_sentences_event_coref_propagated_test.json \
--checkpoint_dir additional_models/coref_pred_propagated_no_proposals/MILO_5_lw_2_v2 \
--obj_feat_dim 2048 --num_candidates 5  --sim_threshold 0.25 \
--train_np_vid_feat_dir /dvmm-filer2/projects/AIDA/data/m2e2_video_features/additional_vids/video_feats \
--n_pair 36 --negative_weighting 0 --max_words 30 --look_window 2 --num_candidates_eval 5
 ``` 
	- best model at additional_models/coref_pred_propagated_no_proposals/MILO_5_lw_2_v2/milo_20.pth

## Multimodal Event Coreference WITH proposals
m2e2_train_milo_eval_coref_with_proposals_sim_threshold.py
 ``` 
CUDA_VISIBLE_DEVICES=6 python m2e2_train_milo_eval_coref_with_proposals_sim_threshold.py \
--m2e2=1 --eval_m2e2=1 --num_thread_reader=16 --batch_size=1 --epochs=15 --n_display=400 --lr_decay=1.0 --embd_dim=6144 \
--pretrain_path=model/howto100m_pt_model.pth \
--videoID2obj_feature_path /dvmm-filer2/projects/AIDA/data/m2e2_video_features/metadata/obj_det_outputs/additional_vids/ten_feats_per_frame/additional_vids2all_ASRs.json \
--video_clip_feature_dir /dvmm-filer2/projects/AIDA/data/m2e2_video_features/original_vids/video_feats \
--m2e2_manual_labels_obj_features_json_val  /dvmm-filer2/projects/AIDA/data/m2e2_video_features/metadata/obj_det_outputs/event_coreference_clips/m2e2_all_event_coreference_pairs_with_entire_duration_features_event_coref_propagated_val.json \
--m2e2_manual_labels_obj_features_json_test  /dvmm-filer2/projects/AIDA/data/m2e2_video_features/metadata/obj_det_outputs/event_coreference_clips/m2e2_all_event_coreference_pairs_with_entire_duration_features_event_coref_propagated_test.json \
--checkpoint_dir additional_models/coref_pred_propagated_with_proposals/MILO_5_lw_2_v2 \
--obj_feat_dim 2048 --num_candidates 5 \
--train_np_vid_feat_dir /dvmm-filer2/projects/AIDA/data/m2e2_video_features/additional_vids/video_feats \
--n_pair 36 --negative_weighting 0 --max_words 30 --look_window 2 --num_candidates_eval 5
 ``` 
	- Best model at additional_models/coref_pred_propagated_with_proposals/MILO_5_lw_2_v2/milo_9.pth

## CLIP baseline for both event coreference with and without proposals
- CLIP Multimodal Event Coreference Baseline.ipynb

## MIL-NCE baseline for both event coreference with and without proposals
- MIL-NCE baseline without Proposals.ipynb
- MIL-NCE baseline with Proposals.ipynb 


## DataLoaders
m2e2_dataloader.py
- M2E2MILOVidSegDiscriminatorWithMultipleSamplingDataLoader  
	- for training
	- same batch negative sampling
	- samples objects from within a window 
	- used in m2e2_train_milo_eval_coref_with_no_proposals.py and m2e2_train_milo_eval_coref_with_proposals_sim_threshold.py
- M2E2MILOManualLabelsWithArticleDataLoader
	- for evaluation without proposals
	- used in m2e2_train_milo_eval_coref_with_no_proposals
- M2E2MILOManualLabelsWithArticleAndProposalsDataLoader
	- for evaluation with proposals
	- used in m2e2_train_milo_eval_coref_with_proposals_sim_threshold.py

## MILO model definition
milo_model.py

## MILO Loss
mil_loss.py 
	- combines MIL-NCE loss and MaxMarginRanking loss from loss.py 

## commandline arguments
args.py

## Qualitative Analysis 
Event Coreference Matching Qualitative Analysis.ipynb 
	- for without proposals
Event Coreference Matching Qualitative Analysis with Proposals.ipynb
	- for with proposals 


# Feature Extraction  
## Global Features
- /home/shoya/video_feature_extractor 
- simply same as https://github.com/antoine77340/video_feature_extractor

## Local Features
- /home/shoya/faster_rcnn
- generate_tsv_additional_vids.py
	- Extracts features every 3 seconds
	- example command below 
 ``` 
CUDA_VISIBLE_DEVICES=0 python generate_tsv_additional_vids.py \
--net res101 --dataset vg --out not_needed.tsv --load_dir data/pretrained_model \
--csv /dvmm-filer2/projects/AIDA/data/m2e2_video_features/metadata/obj_detect_input_files/original_vids_entire_duration/1400_1600.csv \
--vid_dir /dvmm-filer2/projects/AIDA/data/m2e2_videos_original \
--output_dir /dvmm-filer2/projects/AIDA/data/m2e2_video_features/metadata/obj_det_outputs/original_vids/duration_all/feat_per_3_sec \
--cuda \
--np_save_dir /dvmm-filer2/projects/AIDA/data/m2e2_video_features/original_vids/obj_feats/duration_all/feat_per_3_sec/features \
--bbox_save_dir /dvmm-filer2/projects/AIDA/data/m2e2_video_features/original_vids/obj_feats/duration_all/feat_per_3_sec/bbox \
--csv_split_pt_num 1400
 ``` 

# Dataset Collection 
- YouTube Search [notebook](https://colab.research.google.com/drive/1WAf77lCMIOR_XhXw_zCGm0441p7ev8_5?usp=sharing)
	- Input: queries you want to use
	- Output: vid_urls.json
- Downloading Video descriptions (articles) [notebook](https://colab.research.google.com/drive/1bwrK71atOTaWZRk0NXqW80NcBmlrBiid?usp=sharing)
	- Input: vid_urls.json 
	- Output: vid2article.json
- Downloading Captions [notebook](https://colab.research.google.com/drive/1OUlkByu5V2gDEBYpkTmILpwvNjig9DDh?usp=sharing)
	- Input: vid2article.json
	- Output: vid2article_master_not_filtered.json
- Downloading Video files: /home/shoya/AIDA/additional_data_download/download_additional_m2e2_videos.py
	- Input: vid_urls.json
	- Output: downloaded mp4 files
- Clean downloaded files: /home/shoya/AIDA/additional_data_download/Downloaded Data Due Diligence.ipynb
	- Will delete the file entries that were not downloaded correctly from vid2article_master_not_filtered as well as from the output mp4 directory
	- Output: vid2article_master_filtered.json

