export CUDA_VISIBLE_DEVICES=' '

python jpgs2tfrecord_untrimmed.py \
    --hdfs_dir_trimmed /user/VideoAI/rextang/activity_net_full/videos/trimmed/tfrecords \
    --hdfs_dir_untrimmed /user/VideoAI/rextang/activity_net_full/videos/untrimmed/tfrecords \
    --json_path /data1/rextang/activity_net/codes/activity_net/activity_net.v1-3.min.json \
    --video_source /data1/rextang/datasets/activity_net_full \
    --destination ./output_tmp/videos \
    --jpg_path ./images_tmp \
    --FPS 24 \
    --batch_start 0 \
    --batch_end 1 \
    $@