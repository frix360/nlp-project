�)$	��6�4D�?z��p!�?_�Q�[?!$(~��k�?	_oP���?7�d��"�?!��(�k�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$$(~��k�?Zd;�O��?AF%u��?Y�q����o?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/n��b?�~j�t�X?AǺ���F?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails_�Q�[?a2U0*�S?A����Mb@?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsŏ1w-!_?��_�LU?Aa2U0*�C?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsŏ1w-!_?a2U0*�S?AǺ���F?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails_�Q�[?��_�LU?A-C��6:?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/n��b?��_�LU?A��H�}M?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails���_vOn?{�G�zd?Aa2U0*�S?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsŏ1w-!_?�~j�t�X?A-C��6:?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	��_�Le?�~j�t�X?A/n��R?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
ŏ1w-!_?�~j�t�X?A-C��6:?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�J�4q?�����g?A��_�LU?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsHP�s�r?ŏ1w-!o?A-C��6J?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�~j�t�h?��H�}]?Aa2U0*�S?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsŏ1w-!_?�~j�t�X?A-C��6:?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��H�}]?Ǻ���V?A-C��6:?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails_�Q�[?��_�LU?A-C��6:?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails_�Q�[?Ǻ���V?Aa2U0*�3?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsa2U0*�c?ŏ1w-!_?A����Mb@?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsŏ1w-!o?HP�s�b?A�~j�t�X?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsŏ1w-!_?Ǻ���V?A����Mb@?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/n��b?�~j�t�X?AǺ���F?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��H�}m?F%u�k?Aa2U0*�3?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails��H�}]?Ǻ���V?A-C��6:?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsU���N@s?����Mbp?AǺ���F?*	gffff�P@2F
Iterator::Modelh��|?5�?![�xo�%F@)�g��s��?1O}Q���?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?�ܵ�|�?!�⽭�,8@)S�!�uq�?1�31�4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX9��v��?!_��<�E7@) �o_Ή?1�n0E>�2@:Preprocessing2U
Iterator::Model::ParallelMapV2�St$���?!��?���(@)�St$���?1��?���(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��\m���?!�M��T�K@)	�^)�p?1�	?f�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�����g?!��,��j@)�����g?1��,��j@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_vOf?!v�)�Y7@)��_vOf?1v�)�Y7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapX�5�;N�?!���`9@)Ǻ���V?1+�� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 34.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�?�w	�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	QH2�w�m?��}�'�?a2U0*�S?!Zd;�O��?	!       "	!       *	!       2$	J%u�{?2t�BQ�?a2U0*�3?!F%u��?:	!       B	!       J	����}r$?�'�>�I?!�q����o?R	!       Z	����}r$?�'�>�I?!�q����o?JCPU_ONLYY�?�w	�?b Y      Y@q��\B`�?"�
both�Your program is POTENTIALLY input-bound because 34.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 