	������E@������E@!������E@	�I��i\�?�I��i\�?!�I��i\�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$������E@y�ߢ��@ArM��΀C@Yh$B#ظ�?*	�C�l�e�@2F
Iterator::Model�!r�z��?!s{b]$V@)�3�9A��?1�V#%�KU@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat{O崧�?!C��#6�@)|�%j�?1h��é@:Preprocessing2S
Iterator::Model::ParallelMap�߾�3�?!ۮ��g@)�߾�3�?1ۮ��g@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�$��ID�?!��D݋@)�$��ID�?1��D݋@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate~�4bf��?!���̕@)|,}���?1Qp7�@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip>�^����?!h'��&@)� ��zy?1�W��E��?:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�H�5C�?!��FK�@)G6ue?1�ʎXWq�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�[z4�c?!��#��?)�[z4�c?1��#��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*moderate2A8.5 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	y�ߢ��@y�ߢ��@!y�ߢ��@      ��!       "      ��!       *      ��!       2	rM��΀C@rM��΀C@!rM��΀C@:      ��!       B      ��!       J	h$B#ظ�?h$B#ظ�?!h$B#ظ�?R      ��!       Z	h$B#ظ�?h$B#ظ�?!h$B#ظ�?JCPU_ONLY