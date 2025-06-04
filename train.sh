python train.py -s data/multi_object1  \
                --lambda_weight_img_sparse 0.02 \
                --basis_merge_threshold 0.03
               
python train.py -s data/multi_object2  \
                --lambda_weight_img_sparse 0.02 \
                --basis_merge_threshold 0.04

python train.py -s data/multi_object3  \
                --lambda_weight_img_sparse 0.02 \
                --basis_merge_threshold 0.03
                        
python train.py -s data/multi_object4  \
                --lambda_weight_img_sparse 0.01 \
                --basis_merge_threshold 0.015

python train.py -s data/real1  \
                --lambda_weight_sparse 0.01 \
                --basis_merge_threshold 0.015 \
                --gamma
                
python train.py -s data/real2  \
                --lambda_weight_sparse 0.005 \
                --basis_merge_threshold 0.015 \
                --gamma