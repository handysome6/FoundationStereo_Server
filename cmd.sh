python scripts/run_demo.py \
    --left_file /home/andy/DCIM/0327_1/rectified/A_87522380771.jpg \
    --right_file /home/andy/DCIM/0327_1/rectified/D_87522380771_adjusted.jpg \
    --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth \
    --out_dir ./test_outputs/ \
    --scale 0.35 \
    --denoise_cloud 0 

python stereo_client.py \
    --left /home/andy/DCIM/0327_1/rectified/A_87522380771.jpg \
    --right /home/andy/DCIM/0327_1/rectified/D_87522380771_adjusted.jpg \
    --output test_output \
    --intrinsic assets/K_477module.txt \
    --scale 0.35

