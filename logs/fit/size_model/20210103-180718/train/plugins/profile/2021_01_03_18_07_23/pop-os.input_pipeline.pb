$	�b���?�*
��?FCƣT�S?!!��	L��?	'��
��?�7v6@!����1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$!��	L��?5�؀�?A�FN���?YW�f�"�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsap��/�?kg{�?AVa3�ق?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails����Z?G���R{Q?A.2�B?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��D-ͭ`?S�!�uq[?A���1��7?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails\ A�c̭?�~K�|�?Ae����C?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsG6ue?M��ua?Aʩ�ajK=?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsFCƣT�S?I���p�P?A�'�>�)?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsSz��˔?�i�WV��?A1]��a(?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��Q,��Z?㊋�rS?A��N=?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	�බ�?�A��ފ�?Ayxρ�)?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
��	�yk?���1��g?A�/K;5�;?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails46<�Rf?u�b?A�H�[�@?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1�0&��d?������`?A�x#��??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails<P�<�f?b��!��b?A̶�ֈ`<?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsr�#Df?Q�+��b?A���2��;?*	��C��U@2F
Iterator::Model��y�)�?!��fJD@)�_��ME�?1zd��o�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�a��h�?!D�?�<@)��25	ސ?1�����3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�E����?!~sL�H17@)���Q��?1�o:zU0@:Preprocessing2U
Iterator::Model::ParallelMapV2#���?!�� �)�&@)#���?1�� �)�&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���v�
�?!��g�N"@)���v�
�?1��g�N"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�W��Ix?!�OuG9o@)�W��Ix?1�OuG9o@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�����?!��={M@)Y�|^�t?1c����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapӅX��?!�#��8@)�4�;�X?1�i�G��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 12.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t42.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�ve�\\(@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	��3�Պ?�t{1���?I���p�P?!5�؀�?	!       "	!       *	!       2$	4C�/�?��%s;�?1]��a(?!�FN���?:	!       B	!       J	n1�}�o?3	�� �?!W�f�"�?R	!       Z	n1�}�o?3	�� �?!W�f�"�?JCPU_ONLYY�ve�\\(@b 