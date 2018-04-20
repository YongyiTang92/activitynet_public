activity_net 数据目录

Trimmed data:
/user/VideoAI/rextang/activity_net_full/videos/trimmed/tfrecords/training 或 validation 或testing
主要feature：
’images’: 24fps采样的rgb帧，以jpg编码
’flow_x’:24fps采样的rgb帧转灰度计算的flow_x，以jpg编码灰度图，根据相邻帧计算。
’flow_y’:24fps采样的rgb帧转灰度计算的flow_y，以jpg编码灰度图，根据相邻帧计算。
’audio’: 22050fps采样的音频编码，矩阵大小为[22050*seg_duration,1]。minimum second为10
’number_of_frames’:该segment中的帧数
’subset’:’training’ or ‘validation’ or ’testing'
‘resolution’:视频分辨率，为str
‘duration’: 整个video的时间，可能有些没有，对于trimmed data来说不重要
‘label_index’:0开始index的类标
‘label_name’:类标名

UnTrimmed data:
/user/VideoAI/rextang/activity_net_full/videos/untrimmed/tfrecords/training 或 validation 或testing
主要feature：
’images’: 24fps采样的rgb帧，以jpg编码
’flow_x’:24fps采样的rgb帧转灰度计算的flow_x，以jpg编码灰度图，根据相邻帧计算。
’flow_y’:24fps采样的rgb帧转灰度计算的flow_y，以jpg编码灰度图，根据相邻帧计算。
’audio’: 22050fps采样的音频编码，矩阵大小为[22050*duration,1]。minimum second为10
’number_of_frames’:该segment中的帧数
’subset’:’training’ or ‘validation’ or ’testing'
‘resolution’:视频分辨率，为str
‘duration’: 整个video的时间，可能有些没有，对于trimmed data来说不重要
‘label_index’:0开始index的类标，有num_segment个
‘label_name’:类标名，有num_segment个
‘num_segment’:该视频中segment的个数
‘segment':

kinetics 数据目录
/user/VideoAI/rextang/kinetics/videos/trimmed/tfrecords/training 或 validation 或test

Audio 22050Hz sample per second, rescale到-256,256

Flow 根据I3D做改动
https://github.com/deepmind/kinetics-i3d
先将image转到灰度，算TVL1-flow，其range为[-20,20]，再rescale到[0,255]

【踩坑记录】
Errors:
1. HADOOP_HDFS_HOME 设置问题，暂时将HADOOP_HDFS_HOME和HADOOP_HOME设一样
2. libhdfs.so cannot open shared object file no such file or directory，再$HADOOP_HDFS_HOME/lib/native中创建软链 ln -s libhdfs.so.0.0.0 libhdfs.so，然后将$HADOOP_HDFS_HOME/lib/native添加到$LD_LIBRARY_PATH
3. libjvm.so cannot open shared object file no such file or directory，将$JRE_HOME/lib/amd64/server添加到$LD_LIBRARY_PATH
ffmpeg:libXv.so.1 not found： 到别的机器的/usr/lib64 复制libXv.so.1和libXv.so.1.0.0
4. 出现hadoop 在ls的时候 OOM:
export HADOOP_OPTS="-XX:-UseGCOverheadLimit -Xmx16384m"
export HADOOP_CLIENT_OPTS="-XX:-UseGCOverheadLimit -Xmx16384m"

export HADOOP_CLASSPATH="`$HADOOP_HOME/bin/hadoop classpath`"export CLASSPATH="$HADOOP_CLASSPATH" for i in `find ${HADOOP_HOME} -name "*.jar"` do         export CLASSPATH="$i:$CLASSPATH” done
具体的环境参考rextang@100.102.33.4:~/.bashrc
测试例程在rextang@100.102.33.4:~/activity_net/test_code