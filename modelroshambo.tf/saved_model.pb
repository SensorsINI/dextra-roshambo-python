��

��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
�
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.22v2.5.1-97-g957590ea15c8��
~
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer1/kernel
w
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*&
_output_shapes
:*
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
:*
dtype0
~
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer2/kernel
w
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*&
_output_shapes
: *
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
: *
dtype0
~
layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_namelayer3/kernel
w
!layer3/kernel/Read/ReadVariableOpReadVariableOplayer3/kernel*&
_output_shapes
: @*
dtype0
n
layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namelayer3/bias
g
layer3/bias/Read/ReadVariableOpReadVariableOplayer3/bias*
_output_shapes
:@*
dtype0

layer4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*
shared_namelayer4/kernel
x
!layer4/kernel/Read/ReadVariableOpReadVariableOplayer4/kernel*'
_output_shapes
:@�*
dtype0
o
layer4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer4/bias
h
layer4/bias/Read/ReadVariableOpReadVariableOplayer4/bias*
_output_shapes	
:�*
dtype0
�
layer5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*
shared_namelayer5/kernel
y
!layer5/kernel/Read/ReadVariableOpReadVariableOplayer5/kernel*(
_output_shapes
:��*
dtype0
o
layer5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelayer5/bias
h
layer5/bias/Read/ReadVariableOpReadVariableOplayer5/bias*
_output_shapes	
:�*
dtype0
w
layer6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namelayer6/kernel
p
!layer6/kernel/Read/ReadVariableOpReadVariableOplayer6/kernel*
_output_shapes
:	�*
dtype0
n
layer6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer6/bias
g
layer6/bias/Read/ReadVariableOpReadVariableOplayer6/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�+
value�+B�+ B�+
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
R
#	variables
$regularization_losses
%trainable_variables
&	keras_api
h

'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
R
-	variables
.regularization_losses
/trainable_variables
0	keras_api
h

1kernel
2bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
R
7	variables
8regularization_losses
9trainable_variables
:	keras_api
h

;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
R
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
R
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
h

Ikernel
Jbias
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
V
0
1
2
3
'4
(5
16
27
;8
<9
I10
J11
 
V
0
1
2
3
'4
(5
16
27
;8
<9
I10
J11
�
Ometrics
	variables
regularization_losses
Player_regularization_losses
Qlayer_metrics
Rnon_trainable_variables
trainable_variables

Slayers
 
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
Tmetrics
	variables
regularization_losses
Ulayer_regularization_losses
Vlayer_metrics
Wnon_trainable_variables
trainable_variables

Xlayers
 
 
 
�
Ymetrics
	variables
regularization_losses
Zlayer_regularization_losses
[layer_metrics
\non_trainable_variables
trainable_variables

]layers
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
^metrics
	variables
 regularization_losses
_layer_regularization_losses
`layer_metrics
anon_trainable_variables
!trainable_variables

blayers
 
 
 
�
cmetrics
#	variables
$regularization_losses
dlayer_regularization_losses
elayer_metrics
fnon_trainable_variables
%trainable_variables

glayers
YW
VARIABLE_VALUElayer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
�
hmetrics
)	variables
*regularization_losses
ilayer_regularization_losses
jlayer_metrics
knon_trainable_variables
+trainable_variables

llayers
 
 
 
�
mmetrics
-	variables
.regularization_losses
nlayer_regularization_losses
olayer_metrics
pnon_trainable_variables
/trainable_variables

qlayers
YW
VARIABLE_VALUElayer4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
�
rmetrics
3	variables
4regularization_losses
slayer_regularization_losses
tlayer_metrics
unon_trainable_variables
5trainable_variables

vlayers
 
 
 
�
wmetrics
7	variables
8regularization_losses
xlayer_regularization_losses
ylayer_metrics
znon_trainable_variables
9trainable_variables

{layers
YW
VARIABLE_VALUElayer5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
 

;0
<1
�
|metrics
=	variables
>regularization_losses
}layer_regularization_losses
~layer_metrics
non_trainable_variables
?trainable_variables
�layers
 
 
 
�
�metrics
A	variables
Bregularization_losses
 �layer_regularization_losses
�layer_metrics
�non_trainable_variables
Ctrainable_variables
�layers
 
 
 
�
�metrics
E	variables
Fregularization_losses
 �layer_regularization_losses
�layer_metrics
�non_trainable_variables
Gtrainable_variables
�layers
YW
VARIABLE_VALUElayer6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1
 

I0
J1
�
�metrics
K	variables
Lregularization_losses
 �layer_regularization_losses
�layer_metrics
�non_trainable_variables
Mtrainable_variables
�layers
 
 
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_input_1Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1layer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biaslayer5/kernellayer5/biaslayer6/kernellayer6/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_1850
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer3/kernel/Read/ReadVariableOplayer3/bias/Read/ReadVariableOp!layer4/kernel/Read/ReadVariableOplayer4/bias/Read/ReadVariableOp!layer5/kernel/Read/ReadVariableOplayer5/bias/Read/ReadVariableOp!layer6/kernel/Read/ReadVariableOplayer6/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_2204
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biaslayer4/kernellayer4/biaslayer5/kernellayer5/biaslayer6/kernellayer6/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_2250��
�
@
$__inference_pool2_layer_call_fn_1360

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_13542
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
'__inference_roshambo_layer_call_fn_1908

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_roshambo_layer_call_and_return_conditional_losses_16832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
%__inference_layer5_layer_call_fn_2103

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer5_layer_call_and_return_conditional_losses_14862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
@__inference_layer4_layer_call_and_return_conditional_losses_1468

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�1
�
B__inference_roshambo_layer_call_and_return_conditional_losses_1519

inputs%
layer1_1415:
layer1_1417:%
layer2_1433: 
layer2_1435: %
layer3_1451: @
layer3_1453:@&
layer4_1469:@�
layer4_1471:	�'
layer5_1487:��
layer5_1489:	�
layer6_1513:	�
layer6_1515:
identity��layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�layer5/StatefulPartitionedCall�layer6/StatefulPartitionedCall�
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_1415layer1_1417*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer1_layer_call_and_return_conditional_losses_14142 
layer1/StatefulPartitionedCall�
pool1/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_13422
pool1/PartitionedCall�
layer2/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0layer2_1433layer2_1435*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer2_layer_call_and_return_conditional_losses_14322 
layer2/StatefulPartitionedCall�
pool2/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_13542
pool2/PartitionedCall�
layer3/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0layer3_1451layer3_1453*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer3_layer_call_and_return_conditional_losses_14502 
layer3/StatefulPartitionedCall�
pool3/PartitionedCallPartitionedCall'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool3_layer_call_and_return_conditional_losses_13662
pool3/PartitionedCall�
layer4/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0layer4_1469layer4_1471*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer4_layer_call_and_return_conditional_losses_14682 
layer4/StatefulPartitionedCall�
pool4/PartitionedCallPartitionedCall'layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool4_layer_call_and_return_conditional_losses_13782
pool4/PartitionedCall�
layer5/StatefulPartitionedCallStatefulPartitionedCallpool4/PartitionedCall:output:0layer5_1487layer5_1489*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer5_layer_call_and_return_conditional_losses_14862 
layer5/StatefulPartitionedCall�
pool5/PartitionedCallPartitionedCall'layer5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool5_layer_call_and_return_conditional_losses_13902
pool5/PartitionedCall�
flatten/PartitionedCallPartitionedCallpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14992
flatten/PartitionedCall�
layer6/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0layer6_1513layer6_1515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer6_layer_call_and_return_conditional_losses_15122 
layer6/StatefulPartitionedCall�
IdentityIdentity'layer6/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer6/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
@
$__inference_pool5_layer_call_fn_1396

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool5_layer_call_and_return_conditional_losses_13902
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
[
?__inference_pool1_layer_call_and_return_conditional_losses_1342

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
%__inference_layer6_layer_call_fn_2134

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer6_layer_call_and_return_conditional_losses_15122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�D
�	
B__inference_roshambo_layer_call_and_return_conditional_losses_2014

inputs?
%layer1_conv2d_readvariableop_resource:4
&layer1_biasadd_readvariableop_resource:?
%layer2_conv2d_readvariableop_resource: 4
&layer2_biasadd_readvariableop_resource: ?
%layer3_conv2d_readvariableop_resource: @4
&layer3_biasadd_readvariableop_resource:@@
%layer4_conv2d_readvariableop_resource:@�5
&layer4_biasadd_readvariableop_resource:	�A
%layer5_conv2d_readvariableop_resource:��5
&layer5_biasadd_readvariableop_resource:	�8
%layer6_matmul_readvariableop_resource:	�4
&layer6_biasadd_readvariableop_resource:
identity��layer1/BiasAdd/ReadVariableOp�layer1/Conv2D/ReadVariableOp�layer2/BiasAdd/ReadVariableOp�layer2/Conv2D/ReadVariableOp�layer3/BiasAdd/ReadVariableOp�layer3/Conv2D/ReadVariableOp�layer4/BiasAdd/ReadVariableOp�layer4/Conv2D/ReadVariableOp�layer5/BiasAdd/ReadVariableOp�layer5/Conv2D/ReadVariableOp�layer6/BiasAdd/ReadVariableOp�layer6/MatMul/ReadVariableOp�
layer1/Conv2D/ReadVariableOpReadVariableOp%layer1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
layer1/Conv2D/ReadVariableOp�
layer1/Conv2DConv2Dinputs$layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<<*
paddingVALID*
strides
2
layer1/Conv2D�
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer1/BiasAdd/ReadVariableOp�
layer1/BiasAddBiasAddlayer1/Conv2D:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<<2
layer1/BiasAddu
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*/
_output_shapes
:���������<<2
layer1/Relu�
pool1/AvgPoolAvgPoollayer1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
pool1/AvgPool�
layer2/Conv2D/ReadVariableOpReadVariableOp%layer2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
layer2/Conv2D/ReadVariableOp�
layer2/Conv2DConv2Dpool1/AvgPool:output:0$layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
layer2/Conv2D�
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
layer2/BiasAdd/ReadVariableOp�
layer2/BiasAddBiasAddlayer2/Conv2D:output:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
layer2/BiasAddu
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
layer2/Relu�
pool2/AvgPoolAvgPoollayer2/Relu:activations:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
pool2/AvgPool�
layer3/Conv2D/ReadVariableOpReadVariableOp%layer3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
layer3/Conv2D/ReadVariableOp�
layer3/Conv2DConv2Dpool2/AvgPool:output:0$layer3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
layer3/Conv2D�
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
layer3/BiasAdd/ReadVariableOp�
layer3/BiasAddBiasAddlayer3/Conv2D:output:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
layer3/BiasAddu
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
layer3/Relu�
pool3/AvgPoolAvgPoollayer3/Relu:activations:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
pool3/AvgPool�
layer4/Conv2D/ReadVariableOpReadVariableOp%layer4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
layer4/Conv2D/ReadVariableOp�
layer4/Conv2DConv2Dpool3/AvgPool:output:0$layer4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
layer4/Conv2D�
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
layer4/BiasAdd/ReadVariableOp�
layer4/BiasAddBiasAddlayer4/Conv2D:output:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
layer4/BiasAddv
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
layer4/Relu�
pool4/AvgPoolAvgPoollayer4/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
pool4/AvgPool�
layer5/Conv2D/ReadVariableOpReadVariableOp%layer5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
layer5/Conv2D/ReadVariableOp�
layer5/Conv2DConv2Dpool4/AvgPool:output:0$layer5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
layer5/Conv2D�
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
layer5/BiasAdd/ReadVariableOp�
layer5/BiasAddBiasAddlayer5/Conv2D:output:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
layer5/BiasAddv
layer5/ReluRelulayer5/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
layer5/Relu�
pool5/AvgPoolAvgPoollayer5/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
pool5/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten/Const�
flatten/ReshapeReshapepool5/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
layer6/MatMul/ReadVariableOpReadVariableOp%layer6_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
layer6/MatMul/ReadVariableOp�
layer6/MatMulMatMulflatten/Reshape:output:0$layer6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer6/MatMul�
layer6/BiasAdd/ReadVariableOpReadVariableOp&layer6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer6/BiasAdd/ReadVariableOp�
layer6/BiasAddBiasAddlayer6/MatMul:product:0%layer6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer6/BiasAddv
layer6/SoftmaxSoftmaxlayer6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layer6/Softmax�
IdentityIdentitylayer6/Softmax:softmax:0^layer1/BiasAdd/ReadVariableOp^layer1/Conv2D/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/Conv2D/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/Conv2D/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/Conv2D/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/Conv2D/ReadVariableOp^layer6/BiasAdd/ReadVariableOp^layer6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/Conv2D/ReadVariableOplayer1/Conv2D/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/Conv2D/ReadVariableOplayer2/Conv2D/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/Conv2D/ReadVariableOplayer3/Conv2D/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/Conv2D/ReadVariableOplayer4/Conv2D/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/Conv2D/ReadVariableOplayer5/Conv2D/ReadVariableOp2>
layer6/BiasAdd/ReadVariableOplayer6/BiasAdd/ReadVariableOp2<
layer6/MatMul/ReadVariableOplayer6/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
@__inference_layer4_layer_call_and_return_conditional_losses_2094

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�%
�
__inference__traced_save_2204
file_prefix,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer3_kernel_read_readvariableop*
&savev2_layer3_bias_read_readvariableop,
(savev2_layer4_kernel_read_readvariableop*
&savev2_layer4_bias_read_readvariableop,
(savev2_layer5_kernel_read_readvariableop*
&savev2_layer5_bias_read_readvariableop,
(savev2_layer6_kernel_read_readvariableop*
&savev2_layer6_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop(savev2_layer4_kernel_read_readvariableop&savev2_layer4_bias_read_readvariableop(savev2_layer5_kernel_read_readvariableop&savev2_layer5_bias_read_readvariableop(savev2_layer6_kernel_read_readvariableop&savev2_layer6_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::: : : @:@:@�:�:��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:.	*
(
_output_shapes
:��:!


_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�
�
%__inference_layer1_layer_call_fn_2023

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer1_layer_call_and_return_conditional_losses_14142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������<<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
[
?__inference_pool2_layer_call_and_return_conditional_losses_1354

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
[
?__inference_pool4_layer_call_and_return_conditional_losses_1378

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
'__inference_roshambo_layer_call_fn_1739
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_roshambo_layer_call_and_return_conditional_losses_16832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�
�
%__inference_layer2_layer_call_fn_2043

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer2_layer_call_and_return_conditional_losses_14322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�D
�	
B__inference_roshambo_layer_call_and_return_conditional_losses_1961

inputs?
%layer1_conv2d_readvariableop_resource:4
&layer1_biasadd_readvariableop_resource:?
%layer2_conv2d_readvariableop_resource: 4
&layer2_biasadd_readvariableop_resource: ?
%layer3_conv2d_readvariableop_resource: @4
&layer3_biasadd_readvariableop_resource:@@
%layer4_conv2d_readvariableop_resource:@�5
&layer4_biasadd_readvariableop_resource:	�A
%layer5_conv2d_readvariableop_resource:��5
&layer5_biasadd_readvariableop_resource:	�8
%layer6_matmul_readvariableop_resource:	�4
&layer6_biasadd_readvariableop_resource:
identity��layer1/BiasAdd/ReadVariableOp�layer1/Conv2D/ReadVariableOp�layer2/BiasAdd/ReadVariableOp�layer2/Conv2D/ReadVariableOp�layer3/BiasAdd/ReadVariableOp�layer3/Conv2D/ReadVariableOp�layer4/BiasAdd/ReadVariableOp�layer4/Conv2D/ReadVariableOp�layer5/BiasAdd/ReadVariableOp�layer5/Conv2D/ReadVariableOp�layer6/BiasAdd/ReadVariableOp�layer6/MatMul/ReadVariableOp�
layer1/Conv2D/ReadVariableOpReadVariableOp%layer1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
layer1/Conv2D/ReadVariableOp�
layer1/Conv2DConv2Dinputs$layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<<*
paddingVALID*
strides
2
layer1/Conv2D�
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer1/BiasAdd/ReadVariableOp�
layer1/BiasAddBiasAddlayer1/Conv2D:output:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<<2
layer1/BiasAddu
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*/
_output_shapes
:���������<<2
layer1/Relu�
pool1/AvgPoolAvgPoollayer1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
pool1/AvgPool�
layer2/Conv2D/ReadVariableOpReadVariableOp%layer2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
layer2/Conv2D/ReadVariableOp�
layer2/Conv2DConv2Dpool1/AvgPool:output:0$layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
layer2/Conv2D�
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
layer2/BiasAdd/ReadVariableOp�
layer2/BiasAddBiasAddlayer2/Conv2D:output:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
layer2/BiasAddu
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
layer2/Relu�
pool2/AvgPoolAvgPoollayer2/Relu:activations:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
pool2/AvgPool�
layer3/Conv2D/ReadVariableOpReadVariableOp%layer3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
layer3/Conv2D/ReadVariableOp�
layer3/Conv2DConv2Dpool2/AvgPool:output:0$layer3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
layer3/Conv2D�
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
layer3/BiasAdd/ReadVariableOp�
layer3/BiasAddBiasAddlayer3/Conv2D:output:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
layer3/BiasAddu
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
layer3/Relu�
pool3/AvgPoolAvgPoollayer3/Relu:activations:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
pool3/AvgPool�
layer4/Conv2D/ReadVariableOpReadVariableOp%layer4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
layer4/Conv2D/ReadVariableOp�
layer4/Conv2DConv2Dpool3/AvgPool:output:0$layer4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
layer4/Conv2D�
layer4/BiasAdd/ReadVariableOpReadVariableOp&layer4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
layer4/BiasAdd/ReadVariableOp�
layer4/BiasAddBiasAddlayer4/Conv2D:output:0%layer4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
layer4/BiasAddv
layer4/ReluRelulayer4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
layer4/Relu�
pool4/AvgPoolAvgPoollayer4/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
pool4/AvgPool�
layer5/Conv2D/ReadVariableOpReadVariableOp%layer5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
layer5/Conv2D/ReadVariableOp�
layer5/Conv2DConv2Dpool4/AvgPool:output:0$layer5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
layer5/Conv2D�
layer5/BiasAdd/ReadVariableOpReadVariableOp&layer5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
layer5/BiasAdd/ReadVariableOp�
layer5/BiasAddBiasAddlayer5/Conv2D:output:0%layer5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
layer5/BiasAddv
layer5/ReluRelulayer5/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
layer5/Relu�
pool5/AvgPoolAvgPoollayer5/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
pool5/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten/Const�
flatten/ReshapeReshapepool5/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshape�
layer6/MatMul/ReadVariableOpReadVariableOp%layer6_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
layer6/MatMul/ReadVariableOp�
layer6/MatMulMatMulflatten/Reshape:output:0$layer6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer6/MatMul�
layer6/BiasAdd/ReadVariableOpReadVariableOp&layer6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer6/BiasAdd/ReadVariableOp�
layer6/BiasAddBiasAddlayer6/MatMul:product:0%layer6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
layer6/BiasAddv
layer6/SoftmaxSoftmaxlayer6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
layer6/Softmax�
IdentityIdentitylayer6/Softmax:softmax:0^layer1/BiasAdd/ReadVariableOp^layer1/Conv2D/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/Conv2D/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/Conv2D/ReadVariableOp^layer4/BiasAdd/ReadVariableOp^layer4/Conv2D/ReadVariableOp^layer5/BiasAdd/ReadVariableOp^layer5/Conv2D/ReadVariableOp^layer6/BiasAdd/ReadVariableOp^layer6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/Conv2D/ReadVariableOplayer1/Conv2D/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/Conv2D/ReadVariableOplayer2/Conv2D/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/Conv2D/ReadVariableOplayer3/Conv2D/ReadVariableOp2>
layer4/BiasAdd/ReadVariableOplayer4/BiasAdd/ReadVariableOp2<
layer4/Conv2D/ReadVariableOplayer4/Conv2D/ReadVariableOp2>
layer5/BiasAdd/ReadVariableOplayer5/BiasAdd/ReadVariableOp2<
layer5/Conv2D/ReadVariableOplayer5/Conv2D/ReadVariableOp2>
layer6/BiasAdd/ReadVariableOplayer6/BiasAdd/ReadVariableOp2<
layer6/MatMul/ReadVariableOplayer6/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
@__inference_layer5_layer_call_and_return_conditional_losses_1486

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_1499

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
"__inference_signature_wrapper_1850
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_13362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�
�
%__inference_layer4_layer_call_fn_2083

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer4_layer_call_and_return_conditional_losses_14682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
B
&__inference_flatten_layer_call_fn_2119

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14992
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
'__inference_roshambo_layer_call_fn_1879

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_roshambo_layer_call_and_return_conditional_losses_15192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
@__inference_layer5_layer_call_and_return_conditional_losses_2114

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
'__inference_roshambo_layer_call_fn_1546
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_roshambo_layer_call_and_return_conditional_losses_15192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�

�
@__inference_layer6_layer_call_and_return_conditional_losses_1512

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
@__inference_layer2_layer_call_and_return_conditional_losses_2054

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
@__inference_layer6_layer_call_and_return_conditional_losses_2145

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
[
?__inference_pool5_layer_call_and_return_conditional_losses_1390

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�6
�
 __inference__traced_restore_2250
file_prefix8
assignvariableop_layer1_kernel:,
assignvariableop_1_layer1_bias::
 assignvariableop_2_layer2_kernel: ,
assignvariableop_3_layer2_bias: :
 assignvariableop_4_layer3_kernel: @,
assignvariableop_5_layer3_bias:@;
 assignvariableop_6_layer4_kernel:@�-
assignvariableop_7_layer4_bias:	�<
 assignvariableop_8_layer5_kernel:��-
assignvariableop_9_layer5_bias:	�4
!assignvariableop_10_layer6_kernel:	�-
assignvariableop_11_layer6_bias:
identity_13��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp assignvariableop_8_layer5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_layer6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_layer6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12�
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
@__inference_layer3_layer_call_and_return_conditional_losses_2074

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�1
�
B__inference_roshambo_layer_call_and_return_conditional_losses_1819
input_1%
layer1_1782:
layer1_1784:%
layer2_1788: 
layer2_1790: %
layer3_1794: @
layer3_1796:@&
layer4_1800:@�
layer4_1802:	�'
layer5_1806:��
layer5_1808:	�
layer6_1813:	�
layer6_1815:
identity��layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�layer5/StatefulPartitionedCall�layer6/StatefulPartitionedCall�
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_1layer1_1782layer1_1784*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer1_layer_call_and_return_conditional_losses_14142 
layer1/StatefulPartitionedCall�
pool1/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_13422
pool1/PartitionedCall�
layer2/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0layer2_1788layer2_1790*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer2_layer_call_and_return_conditional_losses_14322 
layer2/StatefulPartitionedCall�
pool2/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_13542
pool2/PartitionedCall�
layer3/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0layer3_1794layer3_1796*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer3_layer_call_and_return_conditional_losses_14502 
layer3/StatefulPartitionedCall�
pool3/PartitionedCallPartitionedCall'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool3_layer_call_and_return_conditional_losses_13662
pool3/PartitionedCall�
layer4/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0layer4_1800layer4_1802*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer4_layer_call_and_return_conditional_losses_14682 
layer4/StatefulPartitionedCall�
pool4/PartitionedCallPartitionedCall'layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool4_layer_call_and_return_conditional_losses_13782
pool4/PartitionedCall�
layer5/StatefulPartitionedCallStatefulPartitionedCallpool4/PartitionedCall:output:0layer5_1806layer5_1808*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer5_layer_call_and_return_conditional_losses_14862 
layer5/StatefulPartitionedCall�
pool5/PartitionedCallPartitionedCall'layer5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool5_layer_call_and_return_conditional_losses_13902
pool5/PartitionedCall�
flatten/PartitionedCallPartitionedCallpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14992
flatten/PartitionedCall�
layer6/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0layer6_1813layer6_1815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer6_layer_call_and_return_conditional_losses_15122 
layer6/StatefulPartitionedCall�
IdentityIdentity'layer6/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer6/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�
�
@__inference_layer3_layer_call_and_return_conditional_losses_1450

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
@
$__inference_pool3_layer_call_fn_1372

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool3_layer_call_and_return_conditional_losses_13662
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
@__inference_layer1_layer_call_and_return_conditional_losses_2034

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<<*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������<<2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������<<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
@
$__inference_pool4_layer_call_fn_1384

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool4_layer_call_and_return_conditional_losses_13782
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
@__inference_layer1_layer_call_and_return_conditional_losses_1414

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<<*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<<2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������<<2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������<<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
@__inference_layer2_layer_call_and_return_conditional_losses_1432

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�1
�
B__inference_roshambo_layer_call_and_return_conditional_losses_1779
input_1%
layer1_1742:
layer1_1744:%
layer2_1748: 
layer2_1750: %
layer3_1754: @
layer3_1756:@&
layer4_1760:@�
layer4_1762:	�'
layer5_1766:��
layer5_1768:	�
layer6_1773:	�
layer6_1775:
identity��layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�layer5/StatefulPartitionedCall�layer6/StatefulPartitionedCall�
layer1/StatefulPartitionedCallStatefulPartitionedCallinput_1layer1_1742layer1_1744*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer1_layer_call_and_return_conditional_losses_14142 
layer1/StatefulPartitionedCall�
pool1/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_13422
pool1/PartitionedCall�
layer2/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0layer2_1748layer2_1750*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer2_layer_call_and_return_conditional_losses_14322 
layer2/StatefulPartitionedCall�
pool2/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_13542
pool2/PartitionedCall�
layer3/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0layer3_1754layer3_1756*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer3_layer_call_and_return_conditional_losses_14502 
layer3/StatefulPartitionedCall�
pool3/PartitionedCallPartitionedCall'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool3_layer_call_and_return_conditional_losses_13662
pool3/PartitionedCall�
layer4/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0layer4_1760layer4_1762*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer4_layer_call_and_return_conditional_losses_14682 
layer4/StatefulPartitionedCall�
pool4/PartitionedCallPartitionedCall'layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool4_layer_call_and_return_conditional_losses_13782
pool4/PartitionedCall�
layer5/StatefulPartitionedCallStatefulPartitionedCallpool4/PartitionedCall:output:0layer5_1766layer5_1768*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer5_layer_call_and_return_conditional_losses_14862 
layer5/StatefulPartitionedCall�
pool5/PartitionedCallPartitionedCall'layer5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool5_layer_call_and_return_conditional_losses_13902
pool5/PartitionedCall�
flatten/PartitionedCallPartitionedCallpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14992
flatten/PartitionedCall�
layer6/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0layer6_1773layer6_1775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer6_layer_call_and_return_conditional_losses_15122 
layer6/StatefulPartitionedCall�
IdentityIdentity'layer6/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer6/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�
[
?__inference_pool3_layer_call_and_return_conditional_losses_1366

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�1
�
B__inference_roshambo_layer_call_and_return_conditional_losses_1683

inputs%
layer1_1646:
layer1_1648:%
layer2_1652: 
layer2_1654: %
layer3_1658: @
layer3_1660:@&
layer4_1664:@�
layer4_1666:	�'
layer5_1670:��
layer5_1672:	�
layer6_1677:	�
layer6_1679:
identity��layer1/StatefulPartitionedCall�layer2/StatefulPartitionedCall�layer3/StatefulPartitionedCall�layer4/StatefulPartitionedCall�layer5/StatefulPartitionedCall�layer6/StatefulPartitionedCall�
layer1/StatefulPartitionedCallStatefulPartitionedCallinputslayer1_1646layer1_1648*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer1_layer_call_and_return_conditional_losses_14142 
layer1/StatefulPartitionedCall�
pool1/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_13422
pool1/PartitionedCall�
layer2/StatefulPartitionedCallStatefulPartitionedCallpool1/PartitionedCall:output:0layer2_1652layer2_1654*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer2_layer_call_and_return_conditional_losses_14322 
layer2/StatefulPartitionedCall�
pool2/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool2_layer_call_and_return_conditional_losses_13542
pool2/PartitionedCall�
layer3/StatefulPartitionedCallStatefulPartitionedCallpool2/PartitionedCall:output:0layer3_1658layer3_1660*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer3_layer_call_and_return_conditional_losses_14502 
layer3/StatefulPartitionedCall�
pool3/PartitionedCallPartitionedCall'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool3_layer_call_and_return_conditional_losses_13662
pool3/PartitionedCall�
layer4/StatefulPartitionedCallStatefulPartitionedCallpool3/PartitionedCall:output:0layer4_1664layer4_1666*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer4_layer_call_and_return_conditional_losses_14682 
layer4/StatefulPartitionedCall�
pool4/PartitionedCallPartitionedCall'layer4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool4_layer_call_and_return_conditional_losses_13782
pool4/PartitionedCall�
layer5/StatefulPartitionedCallStatefulPartitionedCallpool4/PartitionedCall:output:0layer5_1670layer5_1672*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer5_layer_call_and_return_conditional_losses_14862 
layer5/StatefulPartitionedCall�
pool5/PartitionedCallPartitionedCall'layer5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool5_layer_call_and_return_conditional_losses_13902
pool5/PartitionedCall�
flatten/PartitionedCallPartitionedCallpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14992
flatten/PartitionedCall�
layer6/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0layer6_1677layer6_1679*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer6_layer_call_and_return_conditional_losses_15122 
layer6/StatefulPartitionedCall�
IdentityIdentity'layer6/StatefulPartitionedCall:output:0^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall^layer3/StatefulPartitionedCall^layer4/StatefulPartitionedCall^layer5/StatefulPartitionedCall^layer6/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2@
layer4/StatefulPartitionedCalllayer4/StatefulPartitionedCall2@
layer5/StatefulPartitionedCalllayer5/StatefulPartitionedCall2@
layer6/StatefulPartitionedCalllayer6/StatefulPartitionedCall:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_2125

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
@
$__inference_pool1_layer_call_fn_1348

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_pool1_layer_call_and_return_conditional_losses_13422
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�Q
�

__inference__wrapped_model_1336
input_1H
.roshambo_layer1_conv2d_readvariableop_resource:=
/roshambo_layer1_biasadd_readvariableop_resource:H
.roshambo_layer2_conv2d_readvariableop_resource: =
/roshambo_layer2_biasadd_readvariableop_resource: H
.roshambo_layer3_conv2d_readvariableop_resource: @=
/roshambo_layer3_biasadd_readvariableop_resource:@I
.roshambo_layer4_conv2d_readvariableop_resource:@�>
/roshambo_layer4_biasadd_readvariableop_resource:	�J
.roshambo_layer5_conv2d_readvariableop_resource:��>
/roshambo_layer5_biasadd_readvariableop_resource:	�A
.roshambo_layer6_matmul_readvariableop_resource:	�=
/roshambo_layer6_biasadd_readvariableop_resource:
identity��&roshambo/layer1/BiasAdd/ReadVariableOp�%roshambo/layer1/Conv2D/ReadVariableOp�&roshambo/layer2/BiasAdd/ReadVariableOp�%roshambo/layer2/Conv2D/ReadVariableOp�&roshambo/layer3/BiasAdd/ReadVariableOp�%roshambo/layer3/Conv2D/ReadVariableOp�&roshambo/layer4/BiasAdd/ReadVariableOp�%roshambo/layer4/Conv2D/ReadVariableOp�&roshambo/layer5/BiasAdd/ReadVariableOp�%roshambo/layer5/Conv2D/ReadVariableOp�&roshambo/layer6/BiasAdd/ReadVariableOp�%roshambo/layer6/MatMul/ReadVariableOp�
%roshambo/layer1/Conv2D/ReadVariableOpReadVariableOp.roshambo_layer1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02'
%roshambo/layer1/Conv2D/ReadVariableOp�
roshambo/layer1/Conv2DConv2Dinput_1-roshambo/layer1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<<*
paddingVALID*
strides
2
roshambo/layer1/Conv2D�
&roshambo/layer1/BiasAdd/ReadVariableOpReadVariableOp/roshambo_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&roshambo/layer1/BiasAdd/ReadVariableOp�
roshambo/layer1/BiasAddBiasAddroshambo/layer1/Conv2D:output:0.roshambo/layer1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������<<2
roshambo/layer1/BiasAdd�
roshambo/layer1/ReluRelu roshambo/layer1/BiasAdd:output:0*
T0*/
_output_shapes
:���������<<2
roshambo/layer1/Relu�
roshambo/pool1/AvgPoolAvgPool"roshambo/layer1/Relu:activations:0*
T0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
roshambo/pool1/AvgPool�
%roshambo/layer2/Conv2D/ReadVariableOpReadVariableOp.roshambo_layer2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02'
%roshambo/layer2/Conv2D/ReadVariableOp�
roshambo/layer2/Conv2DConv2Droshambo/pool1/AvgPool:output:0-roshambo/layer2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
roshambo/layer2/Conv2D�
&roshambo/layer2/BiasAdd/ReadVariableOpReadVariableOp/roshambo_layer2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&roshambo/layer2/BiasAdd/ReadVariableOp�
roshambo/layer2/BiasAddBiasAddroshambo/layer2/Conv2D:output:0.roshambo/layer2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
roshambo/layer2/BiasAdd�
roshambo/layer2/ReluRelu roshambo/layer2/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
roshambo/layer2/Relu�
roshambo/pool2/AvgPoolAvgPool"roshambo/layer2/Relu:activations:0*
T0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
roshambo/pool2/AvgPool�
%roshambo/layer3/Conv2D/ReadVariableOpReadVariableOp.roshambo_layer3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02'
%roshambo/layer3/Conv2D/ReadVariableOp�
roshambo/layer3/Conv2DConv2Droshambo/pool2/AvgPool:output:0-roshambo/layer3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingVALID*
strides
2
roshambo/layer3/Conv2D�
&roshambo/layer3/BiasAdd/ReadVariableOpReadVariableOp/roshambo_layer3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&roshambo/layer3/BiasAdd/ReadVariableOp�
roshambo/layer3/BiasAddBiasAddroshambo/layer3/Conv2D:output:0.roshambo/layer3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
roshambo/layer3/BiasAdd�
roshambo/layer3/ReluRelu roshambo/layer3/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
roshambo/layer3/Relu�
roshambo/pool3/AvgPoolAvgPool"roshambo/layer3/Relu:activations:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
roshambo/pool3/AvgPool�
%roshambo/layer4/Conv2D/ReadVariableOpReadVariableOp.roshambo_layer4_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02'
%roshambo/layer4/Conv2D/ReadVariableOp�
roshambo/layer4/Conv2DConv2Droshambo/pool3/AvgPool:output:0-roshambo/layer4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
roshambo/layer4/Conv2D�
&roshambo/layer4/BiasAdd/ReadVariableOpReadVariableOp/roshambo_layer4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&roshambo/layer4/BiasAdd/ReadVariableOp�
roshambo/layer4/BiasAddBiasAddroshambo/layer4/Conv2D:output:0.roshambo/layer4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
roshambo/layer4/BiasAdd�
roshambo/layer4/ReluRelu roshambo/layer4/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
roshambo/layer4/Relu�
roshambo/pool4/AvgPoolAvgPool"roshambo/layer4/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
roshambo/pool4/AvgPool�
%roshambo/layer5/Conv2D/ReadVariableOpReadVariableOp.roshambo_layer5_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02'
%roshambo/layer5/Conv2D/ReadVariableOp�
roshambo/layer5/Conv2DConv2Droshambo/pool4/AvgPool:output:0-roshambo/layer5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
2
roshambo/layer5/Conv2D�
&roshambo/layer5/BiasAdd/ReadVariableOpReadVariableOp/roshambo_layer5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&roshambo/layer5/BiasAdd/ReadVariableOp�
roshambo/layer5/BiasAddBiasAddroshambo/layer5/Conv2D:output:0.roshambo/layer5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
roshambo/layer5/BiasAdd�
roshambo/layer5/ReluRelu roshambo/layer5/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
roshambo/layer5/Relu�
roshambo/pool5/AvgPoolAvgPool"roshambo/layer5/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
roshambo/pool5/AvgPool�
roshambo/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
roshambo/flatten/Const�
roshambo/flatten/ReshapeReshaperoshambo/pool5/AvgPool:output:0roshambo/flatten/Const:output:0*
T0*(
_output_shapes
:����������2
roshambo/flatten/Reshape�
%roshambo/layer6/MatMul/ReadVariableOpReadVariableOp.roshambo_layer6_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02'
%roshambo/layer6/MatMul/ReadVariableOp�
roshambo/layer6/MatMulMatMul!roshambo/flatten/Reshape:output:0-roshambo/layer6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
roshambo/layer6/MatMul�
&roshambo/layer6/BiasAdd/ReadVariableOpReadVariableOp/roshambo_layer6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&roshambo/layer6/BiasAdd/ReadVariableOp�
roshambo/layer6/BiasAddBiasAdd roshambo/layer6/MatMul:product:0.roshambo/layer6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
roshambo/layer6/BiasAdd�
roshambo/layer6/SoftmaxSoftmax roshambo/layer6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
roshambo/layer6/Softmax�
IdentityIdentity!roshambo/layer6/Softmax:softmax:0'^roshambo/layer1/BiasAdd/ReadVariableOp&^roshambo/layer1/Conv2D/ReadVariableOp'^roshambo/layer2/BiasAdd/ReadVariableOp&^roshambo/layer2/Conv2D/ReadVariableOp'^roshambo/layer3/BiasAdd/ReadVariableOp&^roshambo/layer3/Conv2D/ReadVariableOp'^roshambo/layer4/BiasAdd/ReadVariableOp&^roshambo/layer4/Conv2D/ReadVariableOp'^roshambo/layer5/BiasAdd/ReadVariableOp&^roshambo/layer5/Conv2D/ReadVariableOp'^roshambo/layer6/BiasAdd/ReadVariableOp&^roshambo/layer6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������@@: : : : : : : : : : : : 2P
&roshambo/layer1/BiasAdd/ReadVariableOp&roshambo/layer1/BiasAdd/ReadVariableOp2N
%roshambo/layer1/Conv2D/ReadVariableOp%roshambo/layer1/Conv2D/ReadVariableOp2P
&roshambo/layer2/BiasAdd/ReadVariableOp&roshambo/layer2/BiasAdd/ReadVariableOp2N
%roshambo/layer2/Conv2D/ReadVariableOp%roshambo/layer2/Conv2D/ReadVariableOp2P
&roshambo/layer3/BiasAdd/ReadVariableOp&roshambo/layer3/BiasAdd/ReadVariableOp2N
%roshambo/layer3/Conv2D/ReadVariableOp%roshambo/layer3/Conv2D/ReadVariableOp2P
&roshambo/layer4/BiasAdd/ReadVariableOp&roshambo/layer4/BiasAdd/ReadVariableOp2N
%roshambo/layer4/Conv2D/ReadVariableOp%roshambo/layer4/Conv2D/ReadVariableOp2P
&roshambo/layer5/BiasAdd/ReadVariableOp&roshambo/layer5/BiasAdd/ReadVariableOp2N
%roshambo/layer5/Conv2D/ReadVariableOp%roshambo/layer5/Conv2D/ReadVariableOp2P
&roshambo/layer6/BiasAdd/ReadVariableOp&roshambo/layer6/BiasAdd/ReadVariableOp2N
%roshambo/layer6/MatMul/ReadVariableOp%roshambo/layer6/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������@@
!
_user_specified_name	input_1
�
�
%__inference_layer3_layer_call_fn_2063

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_layer3_layer_call_and_return_conditional_losses_14502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������@@:
layer60
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�r
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
	variables
regularization_losses
trainable_variables
	keras_api

signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"�n
_tf_keras_network�m{"name": "roshambo", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "roshambo", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool1", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2", "inbound_nodes": [[["pool1", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool2", "inbound_nodes": [[["layer2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3", "inbound_nodes": [[["pool2", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool3", "inbound_nodes": [[["layer3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer4", "inbound_nodes": [[["pool3", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool4", "inbound_nodes": [[["layer4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "layer5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer5", "inbound_nodes": [[["pool4", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool5", "inbound_nodes": [[["layer5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["pool5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer6", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer6", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["layer6", 0, 0]]}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 64, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 64, 64, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "roshambo", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "AveragePooling2D", "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool1", "inbound_nodes": [[["layer1", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv2D", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2", "inbound_nodes": [[["pool1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "AveragePooling2D", "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool2", "inbound_nodes": [[["layer2", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3", "inbound_nodes": [[["pool2", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "AveragePooling2D", "config": {"name": "pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool3", "inbound_nodes": [[["layer3", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv2D", "config": {"name": "layer4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer4", "inbound_nodes": [[["pool3", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "AveragePooling2D", "config": {"name": "pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool4", "inbound_nodes": [[["layer4", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Conv2D", "config": {"name": "layer5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer5", "inbound_nodes": [[["pool4", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "AveragePooling2D", "config": {"name": "pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool5", "inbound_nodes": [[["layer5", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["pool5", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Dense", "config": {"name": "layer6", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer6", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 24}], "input_layers": [["input_1", 0, 0]], "output_layers": [["layer6", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"name": "layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 1]}}
�
	variables
regularization_losses
trainable_variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "pool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling2D", "config": {"name": "pool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["layer1", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 28}}
�


kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"name": "layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool1", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 16]}}
�
#	variables
$regularization_losses
%trainable_variables
&	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "pool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling2D", "config": {"name": "pool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["layer2", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 30}}
�


'kernel
(bias
)	variables
*regularization_losses
+trainable_variables
,	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"name": "layer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool2", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
�
-	variables
.regularization_losses
/trainable_variables
0	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "pool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling2D", "config": {"name": "pool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["layer3", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 32}}
�


1kernel
2bias
3	variables
4regularization_losses
5trainable_variables
6	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"name": "layer4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "layer4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool3", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 64]}}
�
7	variables
8regularization_losses
9trainable_variables
:	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "pool4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling2D", "config": {"name": "pool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["layer4", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 34}}
�


;kernel
<bias
=	variables
>regularization_losses
?trainable_variables
@	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�	{"name": "layer5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "layer5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["pool4", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 128]}}
�
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "pool5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AveragePooling2D", "config": {"name": "pool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["layer5", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 36}}
�
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["pool5", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 37}}
�	

Ikernel
Jbias
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "layer6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "layer6", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
v
0
1
2
3
'4
(5
16
27
;8
<9
I10
J11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
'4
(5
16
27
;8
<9
I10
J11"
trackable_list_wrapper
�
Ometrics
	variables
regularization_losses
Player_regularization_losses
Qlayer_metrics
Rnon_trainable_variables
trainable_variables

Slayers
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
':%2layer1/kernel
:2layer1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Tmetrics
	variables
regularization_losses
Ulayer_regularization_losses
Vlayer_metrics
Wnon_trainable_variables
trainable_variables

Xlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ymetrics
	variables
regularization_losses
Zlayer_regularization_losses
[layer_metrics
\non_trainable_variables
trainable_variables

]layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':% 2layer2/kernel
: 2layer2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
^metrics
	variables
 regularization_losses
_layer_regularization_losses
`layer_metrics
anon_trainable_variables
!trainable_variables

blayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
cmetrics
#	variables
$regularization_losses
dlayer_regularization_losses
elayer_metrics
fnon_trainable_variables
%trainable_variables

glayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':% @2layer3/kernel
:@2layer3/bias
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
�
hmetrics
)	variables
*regularization_losses
ilayer_regularization_losses
jlayer_metrics
knon_trainable_variables
+trainable_variables

llayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
mmetrics
-	variables
.regularization_losses
nlayer_regularization_losses
olayer_metrics
pnon_trainable_variables
/trainable_variables

qlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(:&@�2layer4/kernel
:�2layer4/bias
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
�
rmetrics
3	variables
4regularization_losses
slayer_regularization_losses
tlayer_metrics
unon_trainable_variables
5trainable_variables

vlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
wmetrics
7	variables
8regularization_losses
xlayer_regularization_losses
ylayer_metrics
znon_trainable_variables
9trainable_variables

{layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'��2layer5/kernel
:�2layer5/bias
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
�
|metrics
=	variables
>regularization_losses
}layer_regularization_losses
~layer_metrics
non_trainable_variables
?trainable_variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
A	variables
Bregularization_losses
 �layer_regularization_losses
�layer_metrics
�non_trainable_variables
Ctrainable_variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
E	variables
Fregularization_losses
 �layer_regularization_losses
�layer_metrics
�non_trainable_variables
Gtrainable_variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�2layer6/kernel
:2layer6/bias
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
�
�metrics
K	variables
Lregularization_losses
 �layer_regularization_losses
�layer_metrics
�non_trainable_variables
Mtrainable_variables
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
__inference__wrapped_model_1336�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������@@
�2�
'__inference_roshambo_layer_call_fn_1546
'__inference_roshambo_layer_call_fn_1879
'__inference_roshambo_layer_call_fn_1908
'__inference_roshambo_layer_call_fn_1739�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_roshambo_layer_call_and_return_conditional_losses_1961
B__inference_roshambo_layer_call_and_return_conditional_losses_2014
B__inference_roshambo_layer_call_and_return_conditional_losses_1779
B__inference_roshambo_layer_call_and_return_conditional_losses_1819�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_layer1_layer_call_fn_2023�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_layer1_layer_call_and_return_conditional_losses_2034�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_pool1_layer_call_fn_1348�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
?__inference_pool1_layer_call_and_return_conditional_losses_1342�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
%__inference_layer2_layer_call_fn_2043�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_layer2_layer_call_and_return_conditional_losses_2054�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_pool2_layer_call_fn_1360�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
?__inference_pool2_layer_call_and_return_conditional_losses_1354�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
%__inference_layer3_layer_call_fn_2063�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_layer3_layer_call_and_return_conditional_losses_2074�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_pool3_layer_call_fn_1372�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
?__inference_pool3_layer_call_and_return_conditional_losses_1366�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
%__inference_layer4_layer_call_fn_2083�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_layer4_layer_call_and_return_conditional_losses_2094�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_pool4_layer_call_fn_1384�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
?__inference_pool4_layer_call_and_return_conditional_losses_1378�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
%__inference_layer5_layer_call_fn_2103�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_layer5_layer_call_and_return_conditional_losses_2114�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_pool5_layer_call_fn_1396�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
?__inference_pool5_layer_call_and_return_conditional_losses_1390�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
&__inference_flatten_layer_call_fn_2119�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_flatten_layer_call_and_return_conditional_losses_2125�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_layer6_layer_call_fn_2134�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_layer6_layer_call_and_return_conditional_losses_2145�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_1850input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference__wrapped_model_1336y'(12;<IJ8�5
.�+
)�&
input_1���������@@
� "/�,
*
layer6 �
layer6����������
A__inference_flatten_layer_call_and_return_conditional_losses_2125b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� 
&__inference_flatten_layer_call_fn_2119U8�5
.�+
)�&
inputs����������
� "������������
@__inference_layer1_layer_call_and_return_conditional_losses_2034l7�4
-�*
(�%
inputs���������@@
� "-�*
#� 
0���������<<
� �
%__inference_layer1_layer_call_fn_2023_7�4
-�*
(�%
inputs���������@@
� " ����������<<�
@__inference_layer2_layer_call_and_return_conditional_losses_2054l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
%__inference_layer2_layer_call_fn_2043_7�4
-�*
(�%
inputs���������
� " ���������� �
@__inference_layer3_layer_call_and_return_conditional_losses_2074l'(7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
%__inference_layer3_layer_call_fn_2063_'(7�4
-�*
(�%
inputs��������� 
� " ����������@�
@__inference_layer4_layer_call_and_return_conditional_losses_2094m127�4
-�*
(�%
inputs���������@
� ".�+
$�!
0����������
� �
%__inference_layer4_layer_call_fn_2083`127�4
-�*
(�%
inputs���������@
� "!������������
@__inference_layer5_layer_call_and_return_conditional_losses_2114n;<8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
%__inference_layer5_layer_call_fn_2103a;<8�5
.�+
)�&
inputs����������
� "!������������
@__inference_layer6_layer_call_and_return_conditional_losses_2145]IJ0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� y
%__inference_layer6_layer_call_fn_2134PIJ0�-
&�#
!�
inputs����������
� "�����������
?__inference_pool1_layer_call_and_return_conditional_losses_1342�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
$__inference_pool1_layer_call_fn_1348�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
?__inference_pool2_layer_call_and_return_conditional_losses_1354�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
$__inference_pool2_layer_call_fn_1360�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
?__inference_pool3_layer_call_and_return_conditional_losses_1366�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
$__inference_pool3_layer_call_fn_1372�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
?__inference_pool4_layer_call_and_return_conditional_losses_1378�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
$__inference_pool4_layer_call_fn_1384�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
?__inference_pool5_layer_call_and_return_conditional_losses_1390�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
$__inference_pool5_layer_call_fn_1396�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
B__inference_roshambo_layer_call_and_return_conditional_losses_1779w'(12;<IJ@�=
6�3
)�&
input_1���������@@
p 

 
� "%�"
�
0���������
� �
B__inference_roshambo_layer_call_and_return_conditional_losses_1819w'(12;<IJ@�=
6�3
)�&
input_1���������@@
p

 
� "%�"
�
0���������
� �
B__inference_roshambo_layer_call_and_return_conditional_losses_1961v'(12;<IJ?�<
5�2
(�%
inputs���������@@
p 

 
� "%�"
�
0���������
� �
B__inference_roshambo_layer_call_and_return_conditional_losses_2014v'(12;<IJ?�<
5�2
(�%
inputs���������@@
p

 
� "%�"
�
0���������
� �
'__inference_roshambo_layer_call_fn_1546j'(12;<IJ@�=
6�3
)�&
input_1���������@@
p 

 
� "�����������
'__inference_roshambo_layer_call_fn_1739j'(12;<IJ@�=
6�3
)�&
input_1���������@@
p

 
� "�����������
'__inference_roshambo_layer_call_fn_1879i'(12;<IJ?�<
5�2
(�%
inputs���������@@
p 

 
� "�����������
'__inference_roshambo_layer_call_fn_1908i'(12;<IJ?�<
5�2
(�%
inputs���������@@
p

 
� "�����������
"__inference_signature_wrapper_1850�'(12;<IJC�@
� 
9�6
4
input_1)�&
input_1���������@@"/�,
*
layer6 �
layer6���������