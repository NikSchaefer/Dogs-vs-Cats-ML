??!
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
|
training/Adamax/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining/Adamax/iter
u
(training/Adamax/iter/Read/ReadVariableOpReadVariableOptraining/Adamax/iter*
_output_shapes
: *
dtype0	
?
training/Adamax/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining/Adamax/beta_1
y
*training/Adamax/beta_1/Read/ReadVariableOpReadVariableOptraining/Adamax/beta_1*
_output_shapes
: *
dtype0
?
training/Adamax/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining/Adamax/beta_2
y
*training/Adamax/beta_2/Read/ReadVariableOpReadVariableOptraining/Adamax/beta_2*
_output_shapes
: *
dtype0
~
training/Adamax/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining/Adamax/decay
w
)training/Adamax/decay/Read/ReadVariableOpReadVariableOptraining/Adamax/decay*
_output_shapes
: *
dtype0
?
training/Adamax/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining/Adamax/learning_rate
?
1training/Adamax/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adamax/learning_rate*
_output_shapes
: *
dtype0
?
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
?
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
?
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
?
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
?
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameblock2_conv1/kernel
?
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@?*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:?*
dtype0
?
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock2_conv2/kernel
?
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv1/kernel
?
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:?*
dtype0
?
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv2/kernel
?
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv3/kernel
?
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:?*
dtype0
?
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv1/kernel
?
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:?*
dtype0
?
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv2/kernel
?
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:?*
dtype0
?
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv3/kernel
?
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:?*
dtype0
?
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv1/kernel
?
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:?*
dtype0
?
block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv2/kernel
?
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:?*
dtype0
?
block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv3/kernel
?
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:?*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
training/Adamax/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name training/Adamax/dense/kernel/m
?
2training/Adamax/dense/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense/kernel/m* 
_output_shapes
:
??*
dtype0
?
training/Adamax/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nametraining/Adamax/dense/bias/m
?
0training/Adamax/dense/bias/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense/bias/m*
_output_shapes	
:?*
dtype0
?
 training/Adamax/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" training/Adamax/dense_1/kernel/m
?
4training/Adamax/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp training/Adamax/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0
?
training/Adamax/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name training/Adamax/dense_1/bias/m
?
2training/Adamax/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
 training/Adamax/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" training/Adamax/dense_2/kernel/m
?
4training/Adamax/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp training/Adamax/dense_2/kernel/m* 
_output_shapes
:
??*
dtype0
?
training/Adamax/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name training/Adamax/dense_2/bias/m
?
2training/Adamax/dense_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_2/bias/m*
_output_shapes	
:?*
dtype0
?
 training/Adamax/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" training/Adamax/dense_3/kernel/m
?
4training/Adamax/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp training/Adamax/dense_3/kernel/m*
_output_shapes
:	?*
dtype0
?
training/Adamax/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training/Adamax/dense_3/bias/m
?
2training/Adamax/dense_3/bias/m/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_3/bias/m*
_output_shapes
:*
dtype0
?
training/Adamax/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*/
shared_name training/Adamax/dense/kernel/v
?
2training/Adamax/dense/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense/kernel/v* 
_output_shapes
:
??*
dtype0
?
training/Adamax/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_nametraining/Adamax/dense/bias/v
?
0training/Adamax/dense/bias/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense/bias/v*
_output_shapes	
:?*
dtype0
?
 training/Adamax/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" training/Adamax/dense_1/kernel/v
?
4training/Adamax/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp training/Adamax/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0
?
training/Adamax/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name training/Adamax/dense_1/bias/v
?
2training/Adamax/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_1/bias/v*
_output_shapes	
:?*
dtype0
?
 training/Adamax/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*1
shared_name" training/Adamax/dense_2/kernel/v
?
4training/Adamax/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp training/Adamax/dense_2/kernel/v* 
_output_shapes
:
??*
dtype0
?
training/Adamax/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name training/Adamax/dense_2/bias/v
?
2training/Adamax/dense_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_2/bias/v*
_output_shapes	
:?*
dtype0
?
 training/Adamax/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" training/Adamax/dense_3/kernel/v
?
4training/Adamax/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp training/Adamax/dense_3/kernel/v*
_output_shapes
:	?*
dtype0
?
training/Adamax/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training/Adamax/dense_3/bias/v
?
2training/Adamax/dense_3/bias/v/Read/ReadVariableOpReadVariableOptraining/Adamax/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ւ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
 
?

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer-18
layer-19
	variables
regularization_losses
 trainable_variables
!	keras_api
?
"layer_with_weights-0
"layer-0
#layer_with_weights-1
#layer-1
$layer-2
%layer_with_weights-2
%layer-3
&layer_with_weights-3
&layer-4
	optimizer
'	variables
(regularization_losses
)trainable_variables
*	keras_api
?
+iter

,beta_1

-beta_2
	.decay
/learning_rateJm?Km?Lm?Mm?Nm?Om?Pm?Qm?Jv?Kv?Lv?Mv?Nv?Ov?Pv?Qv?
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23
H24
I25
J26
K27
L28
M29
N30
O31
P32
Q33
 
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23
H24
I25
J26
K27
L28
M29
N30
O31
P32
Q33
?
Rnon_trainable_variables
	variables
Smetrics
regularization_losses

Tlayers
trainable_variables
Ulayer_regularization_losses
Vlayer_metrics
 
 
h

0kernel
1bias
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
h

2kernel
3bias
[	variables
\regularization_losses
]trainable_variables
^	keras_api
R
_	variables
`regularization_losses
atrainable_variables
b	keras_api
h

4kernel
5bias
c	variables
dregularization_losses
etrainable_variables
f	keras_api
h

6kernel
7bias
g	variables
hregularization_losses
itrainable_variables
j	keras_api
R
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
h

8kernel
9bias
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
h

:kernel
;bias
s	variables
tregularization_losses
utrainable_variables
v	keras_api
h

<kernel
=bias
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
R
{	variables
|regularization_losses
}trainable_variables
~	keras_api
k

>kernel
?bias
	variables
?regularization_losses
?trainable_variables
?	keras_api
l

@kernel
Abias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Bkernel
Cbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Dkernel
Ebias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Fkernel
Gbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Hkernel
Ibias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23
H24
I25
 
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23
H24
I25
?
?non_trainable_variables
	variables
?metrics
regularization_losses
?layers
 trainable_variables
 ?layer_regularization_losses
?layer_metrics
l

Jkernel
Kbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Lkernel
Mbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Nkernel
Obias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Pkernel
Qbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
8
J0
K1
L2
M3
N4
O5
P6
Q7
 
8
J0
K1
L2
M3
N4
O5
P6
Q7
?
?non_trainable_variables
'	variables
?metrics
(regularization_losses
?layers
)trainable_variables
 ?layer_regularization_losses
?layer_metrics
SQ
VARIABLE_VALUEtraining/Adamax/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining/Adamax/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining/Adamax/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adamax/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining/Adamax/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock4_conv3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock4_conv3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEblock5_conv3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEblock5_conv3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUE
dense/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_1/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_2/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_2/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_3/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_3/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
 

?0

0
1
2
 
 

00
11
 

00
11
?
?non_trainable_variables
W	variables
?metrics
Xregularization_losses
?layers
Ytrainable_variables
 ?layer_regularization_losses
?layer_metrics

20
31
 

20
31
?
?non_trainable_variables
[	variables
?metrics
\regularization_losses
?layers
]trainable_variables
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
?non_trainable_variables
_	variables
?metrics
`regularization_losses
?layers
atrainable_variables
 ?layer_regularization_losses
?layer_metrics

40
51
 

40
51
?
?non_trainable_variables
c	variables
?metrics
dregularization_losses
?layers
etrainable_variables
 ?layer_regularization_losses
?layer_metrics

60
71
 

60
71
?
?non_trainable_variables
g	variables
?metrics
hregularization_losses
?layers
itrainable_variables
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
?non_trainable_variables
k	variables
?metrics
lregularization_losses
?layers
mtrainable_variables
 ?layer_regularization_losses
?layer_metrics

80
91
 

80
91
?
?non_trainable_variables
o	variables
?metrics
pregularization_losses
?layers
qtrainable_variables
 ?layer_regularization_losses
?layer_metrics

:0
;1
 

:0
;1
?
?non_trainable_variables
s	variables
?metrics
tregularization_losses
?layers
utrainable_variables
 ?layer_regularization_losses
?layer_metrics

<0
=1
 

<0
=1
?
?non_trainable_variables
w	variables
?metrics
xregularization_losses
?layers
ytrainable_variables
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
?non_trainable_variables
{	variables
?metrics
|regularization_losses
?layers
}trainable_variables
 ?layer_regularization_losses
?layer_metrics

>0
?1
 

>0
?1
?
?non_trainable_variables
	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics

@0
A1
 

@0
A1
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics

B0
C1
 

B0
C1
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics

D0
E1
 

D0
E1
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics

F0
G1
 

F0
G1
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics

H0
I1
 

H0
I1
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
 
 
?

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
 
 

J0
K1
 

J0
K1
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics

L0
M1
 

L0
M1
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics

N0
O1
 

N0
O1
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics

P0
Q1
 

P0
Q1
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
 

?0
#
"0
#1
$2
%3
&4
 
 
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
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
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
QO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
db
VARIABLE_VALUEtotalIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEcountIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
wu
VARIABLE_VALUEtraining/Adamax/dense/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEtraining/Adamax/dense/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE training/Adamax/dense_1/kernel/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining/Adamax/dense_1/bias/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE training/Adamax/dense_2/kernel/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining/Adamax/dense_2/bias/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE training/Adamax/dense_3/kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining/Adamax/dense_3/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining/Adamax/dense/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEtraining/Adamax/dense/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE training/Adamax/dense_1/kernel/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining/Adamax/dense_1/bias/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE training/Adamax/dense_2/kernel/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining/Adamax/dense_2/bias/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE training/Adamax/dense_3/kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEtraining/Adamax/dense_3/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_2Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_2395
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(training/Adamax/iter/Read/ReadVariableOp*training/Adamax/beta_1/Read/ReadVariableOp*training/Adamax/beta_2/Read/ReadVariableOp)training/Adamax/decay/Read/ReadVariableOp1training/Adamax/learning_rate/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2training/Adamax/dense/kernel/m/Read/ReadVariableOp0training/Adamax/dense/bias/m/Read/ReadVariableOp4training/Adamax/dense_1/kernel/m/Read/ReadVariableOp2training/Adamax/dense_1/bias/m/Read/ReadVariableOp4training/Adamax/dense_2/kernel/m/Read/ReadVariableOp2training/Adamax/dense_2/bias/m/Read/ReadVariableOp4training/Adamax/dense_3/kernel/m/Read/ReadVariableOp2training/Adamax/dense_3/bias/m/Read/ReadVariableOp2training/Adamax/dense/kernel/v/Read/ReadVariableOp0training/Adamax/dense/bias/v/Read/ReadVariableOp4training/Adamax/dense_1/kernel/v/Read/ReadVariableOp2training/Adamax/dense_1/bias/v/Read/ReadVariableOp4training/Adamax/dense_2/kernel/v/Read/ReadVariableOp2training/Adamax/dense_2/bias/v/Read/ReadVariableOp4training/Adamax/dense_3/kernel/v/Read/ReadVariableOp2training/Adamax/dense_3/bias/v/Read/ReadVariableOpConst*H
TinA
?2=	*
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
GPU 2J 8? *&
f!R
__inference__traced_save_3635
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametraining/Adamax/itertraining/Adamax/beta_1training/Adamax/beta_2training/Adamax/decaytraining/Adamax/learning_rateblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biastotal_1count_1totalcounttraining/Adamax/dense/kernel/mtraining/Adamax/dense/bias/m training/Adamax/dense_1/kernel/mtraining/Adamax/dense_1/bias/m training/Adamax/dense_2/kernel/mtraining/Adamax/dense_2/bias/m training/Adamax/dense_3/kernel/mtraining/Adamax/dense_3/bias/mtraining/Adamax/dense/kernel/vtraining/Adamax/dense/bias/v training/Adamax/dense_1/kernel/vtraining/Adamax/dense_1/bias/v training/Adamax/dense_2/kernel/vtraining/Adamax/dense_2/bias/v training/Adamax/dense_3/kernel/vtraining/Adamax/dense_3/bias/v*G
Tin@
>2<*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_3822??
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_1884

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?	
?__inference_model_layer_call_and_return_conditional_losses_2238

inputs
vgg16_block1_conv1_kernel
vgg16_block1_conv1_bias
vgg16_block1_conv2_kernel
vgg16_block1_conv2_bias
vgg16_block2_conv1_kernel
vgg16_block2_conv1_bias
vgg16_block2_conv2_kernel
vgg16_block2_conv2_bias
vgg16_block3_conv1_kernel
vgg16_block3_conv1_bias
vgg16_block3_conv2_kernel
vgg16_block3_conv2_bias
vgg16_block3_conv3_kernel
vgg16_block3_conv3_bias
vgg16_block4_conv1_kernel
vgg16_block4_conv1_bias
vgg16_block4_conv2_kernel
vgg16_block4_conv2_bias
vgg16_block4_conv3_kernel
vgg16_block4_conv3_bias
vgg16_block5_conv1_kernel
vgg16_block5_conv1_bias
vgg16_block5_conv2_kernel
vgg16_block5_conv2_bias
vgg16_block5_conv3_kernel
vgg16_block5_conv3_bias
sequential_dense_kernel
sequential_dense_bias
sequential_dense_1_kernel
sequential_dense_1_bias
sequential_dense_2_kernel
sequential_dense_2_bias
sequential_dense_3_kernel
sequential_dense_3_bias
identity??"sequential/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?
vgg16/StatefulPartitionedCallStatefulPartitionedCallinputsvgg16_block1_conv1_kernelvgg16_block1_conv1_biasvgg16_block1_conv2_kernelvgg16_block1_conv2_biasvgg16_block2_conv1_kernelvgg16_block2_conv1_biasvgg16_block2_conv2_kernelvgg16_block2_conv2_biasvgg16_block3_conv1_kernelvgg16_block3_conv1_biasvgg16_block3_conv2_kernelvgg16_block3_conv2_biasvgg16_block3_conv3_kernelvgg16_block3_conv3_biasvgg16_block4_conv1_kernelvgg16_block4_conv1_biasvgg16_block4_conv2_kernelvgg16_block4_conv2_biasvgg16_block4_conv3_kernelvgg16_block4_conv3_biasvgg16_block5_conv1_kernelvgg16_block5_conv1_biasvgg16_block5_conv2_kernelvgg16_block5_conv2_biasvgg16_block5_conv3_kernelvgg16_block5_conv3_bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_17122
vgg16/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0sequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_biassequential_dense_3_kernelsequential_dense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_19852$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_1181

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_3103

inputs
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_20152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_block1_conv1_layer_call_fn_3121

inputs
block1_conv1_kernel
block1_conv1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_kernelblock1_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_12862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_dense_1_layer_call_fn_3373

inputs
dense_1_kernel
dense_1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_kerneldense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_18592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_1889

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv1_layer_call_and_return_conditional_losses_1540

inputs-
)conv2d_readvariableop_block5_conv1_kernel,
(biasadd_readvariableop_block5_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_1173

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_block3_conv1_layer_call_fn_3193

inputs
block3_conv1_kernel
block3_conv1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock3_conv1_kernelblock3_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_13902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????88?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
+__inference_block4_conv2_layer_call_fn_3265

inputs
block4_conv2_kernel
block4_conv2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock4_conv2_kernelblock4_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_14882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1268

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block3_conv3_layer_call_and_return_conditional_losses_1436

inputs-
)conv2d_readvariableop_block3_conv3_kernel,
(biasadd_readvariableop_block3_conv3_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????88?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_3348

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block3_conv3_layer_call_and_return_conditional_losses_3222

inputs-
)conv2d_readvariableop_block3_conv3_kernel,
(biasadd_readvariableop_block3_conv3_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????88?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
F__inference_block3_conv1_layer_call_and_return_conditional_losses_3186

inputs-
)conv2d_readvariableop_block3_conv1_kernel,
(biasadd_readvariableop_block3_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????88?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_3822
file_prefix)
%assignvariableop_training_adamax_iter-
)assignvariableop_1_training_adamax_beta_1-
)assignvariableop_2_training_adamax_beta_2,
(assignvariableop_3_training_adamax_decay4
0assignvariableop_4_training_adamax_learning_rate*
&assignvariableop_5_block1_conv1_kernel(
$assignvariableop_6_block1_conv1_bias*
&assignvariableop_7_block1_conv2_kernel(
$assignvariableop_8_block1_conv2_bias*
&assignvariableop_9_block2_conv1_kernel)
%assignvariableop_10_block2_conv1_bias+
'assignvariableop_11_block2_conv2_kernel)
%assignvariableop_12_block2_conv2_bias+
'assignvariableop_13_block3_conv1_kernel)
%assignvariableop_14_block3_conv1_bias+
'assignvariableop_15_block3_conv2_kernel)
%assignvariableop_16_block3_conv2_bias+
'assignvariableop_17_block3_conv3_kernel)
%assignvariableop_18_block3_conv3_bias+
'assignvariableop_19_block4_conv1_kernel)
%assignvariableop_20_block4_conv1_bias+
'assignvariableop_21_block4_conv2_kernel)
%assignvariableop_22_block4_conv2_bias+
'assignvariableop_23_block4_conv3_kernel)
%assignvariableop_24_block4_conv3_bias+
'assignvariableop_25_block5_conv1_kernel)
%assignvariableop_26_block5_conv1_bias+
'assignvariableop_27_block5_conv2_kernel)
%assignvariableop_28_block5_conv2_bias+
'assignvariableop_29_block5_conv3_kernel)
%assignvariableop_30_block5_conv3_bias$
 assignvariableop_31_dense_kernel"
assignvariableop_32_dense_bias&
"assignvariableop_33_dense_1_kernel$
 assignvariableop_34_dense_1_bias&
"assignvariableop_35_dense_2_kernel$
 assignvariableop_36_dense_2_bias&
"assignvariableop_37_dense_3_kernel$
 assignvariableop_38_dense_3_bias
assignvariableop_39_total_1
assignvariableop_40_count_1
assignvariableop_41_total
assignvariableop_42_count6
2assignvariableop_43_training_adamax_dense_kernel_m4
0assignvariableop_44_training_adamax_dense_bias_m8
4assignvariableop_45_training_adamax_dense_1_kernel_m6
2assignvariableop_46_training_adamax_dense_1_bias_m8
4assignvariableop_47_training_adamax_dense_2_kernel_m6
2assignvariableop_48_training_adamax_dense_2_bias_m8
4assignvariableop_49_training_adamax_dense_3_kernel_m6
2assignvariableop_50_training_adamax_dense_3_bias_m6
2assignvariableop_51_training_adamax_dense_kernel_v4
0assignvariableop_52_training_adamax_dense_bias_v8
4assignvariableop_53_training_adamax_dense_1_kernel_v6
2assignvariableop_54_training_adamax_dense_1_bias_v8
4assignvariableop_55_training_adamax_dense_2_kernel_v6
2assignvariableop_56_training_adamax_dense_2_bias_v8
4assignvariableop_57_training_adamax_dense_3_kernel_v6
2assignvariableop_58_training_adamax_dense_3_bias_v
identity_60??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*J
dtypes@
>2<	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp%assignvariableop_training_adamax_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_training_adamax_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp)assignvariableop_2_training_adamax_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_training_adamax_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp0assignvariableop_4_training_adamax_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp&assignvariableop_5_block1_conv1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_block1_conv1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_block1_conv2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_block1_conv2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp&assignvariableop_9_block2_conv1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp%assignvariableop_10_block2_conv1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_block2_conv2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_block2_conv2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_block3_conv1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_block3_conv1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_block3_conv2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_block3_conv2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_block3_conv3_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_block3_conv3_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_block4_conv1_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_block4_conv1_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_block4_conv2_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_block4_conv2_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_block4_conv3_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_block4_conv3_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_block5_conv1_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_block5_conv1_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_block5_conv2_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_block5_conv2_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_block5_conv3_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_block5_conv3_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp assignvariableop_31_dense_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_dense_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp"assignvariableop_33_dense_1_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp assignvariableop_34_dense_1_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp"assignvariableop_35_dense_2_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp assignvariableop_36_dense_2_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp"assignvariableop_37_dense_3_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp assignvariableop_38_dense_3_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp2assignvariableop_43_training_adamax_dense_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp0assignvariableop_44_training_adamax_dense_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp4assignvariableop_45_training_adamax_dense_1_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp2assignvariableop_46_training_adamax_dense_1_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp4assignvariableop_47_training_adamax_dense_2_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp2assignvariableop_48_training_adamax_dense_2_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp4assignvariableop_49_training_adamax_dense_3_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp2assignvariableop_50_training_adamax_dense_3_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp2assignvariableop_51_training_adamax_dense_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp0assignvariableop_52_training_adamax_dense_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp4assignvariableop_53_training_adamax_dense_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp2assignvariableop_54_training_adamax_dense_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp4assignvariableop_55_training_adamax_dense_2_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp2assignvariableop_56_training_adamax_dense_2_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp4assignvariableop_57_training_adamax_dense_3_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp2assignvariableop_58_training_adamax_dense_3_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_589
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_59?

Identity_60IdentityIdentity_59:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_60"#
identity_60Identity_60:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_1912

inputs(
$matmul_readvariableop_dense_2_kernel'
#biasadd_readvariableop_dense_2_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_1859

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_2701

inputs
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias
block5_conv2_kernel
block5_conv2_bias
block5_conv3_kernel
block5_conv3_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biasblock5_conv2_kernelblock5_conv2_biasblock5_conv3_kernelblock5_conv3_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_22382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv2_layer_call_and_return_conditional_losses_1488

inputs-
)conv2d_readvariableop_block4_conv2_kernel,
(biasadd_readvariableop_block4_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_3132

inputs-
)conv2d_readvariableop_block1_conv2_kernel,
(biasadd_readvariableop_block1_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
+__inference_block2_conv2_layer_call_fn_3175

inputs
block2_conv2_kernel
block2_conv2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock2_conv2_kernelblock2_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_13612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????pp?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?
F
*__inference_block2_pool_layer_call_fn_1201

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_11982
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_2015

inputs
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
dense_3_dense_3_kernel
dense_3_dense_3_bias
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_18362
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_18592!
dense_1/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_18892
dropout/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_19122!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_19352!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block3_conv1_layer_call_and_return_conditional_losses_1390

inputs-
)conv2d_readvariableop_block3_conv1_kernel,
(biasadd_readvariableop_block3_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????88?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1985

inputs
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
dense_3_dense_3_kernel
dense_3_dense_3_bias
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_18362
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_18592!
dense_1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_18842!
dropout/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_19122!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_19352!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_block5_conv3_layer_call_fn_3337

inputs
block5_conv3_kernel
block5_conv3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock5_conv3_kernelblock5_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_15862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_vgg16_layer_call_fn_2975

inputs
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias
block5_conv2_kernel
block5_conv2_bias
block5_conv3_kernel
block5_conv3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biasblock5_conv2_kernelblock5_conv2_biasblock5_conv3_kernelblock5_conv3_bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_17122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_vgg16_layer_call_fn_3006

inputs
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias
block5_conv2_kernel
block5_conv2_bias
block5_conv3_kernel
block5_conv3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biasblock5_conv2_kernelblock5_conv2_biasblock5_conv3_kernelblock5_conv3_bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_17922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_block1_pool_layer_call_fn_1184

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_11812
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?`
?
?__inference_vgg16_layer_call_and_return_conditional_losses_1611
input_1$
 block1_conv1_block1_conv1_kernel"
block1_conv1_block1_conv1_bias$
 block1_conv2_block1_conv2_kernel"
block1_conv2_block1_conv2_bias$
 block2_conv1_block2_conv1_kernel"
block2_conv1_block2_conv1_bias$
 block2_conv2_block2_conv2_kernel"
block2_conv2_block2_conv2_bias$
 block3_conv1_block3_conv1_kernel"
block3_conv1_block3_conv1_bias$
 block3_conv2_block3_conv2_kernel"
block3_conv2_block3_conv2_bias$
 block3_conv3_block3_conv3_kernel"
block3_conv3_block3_conv3_bias$
 block4_conv1_block4_conv1_kernel"
block4_conv1_block4_conv1_bias$
 block4_conv2_block4_conv2_kernel"
block4_conv2_block4_conv2_bias$
 block4_conv3_block4_conv3_kernel"
block4_conv3_block4_conv3_bias$
 block5_conv1_block5_conv1_kernel"
block5_conv1_block5_conv1_bias$
 block5_conv2_block5_conv2_kernel"
block5_conv2_block5_conv2_bias$
 block5_conv3_block5_conv3_kernel"
block5_conv3_block5_conv3_bias
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1 block1_conv1_block1_conv1_kernelblock1_conv1_block1_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_12862&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0 block1_conv2_block1_conv2_kernelblock1_conv2_block1_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_13092&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_11812
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0 block2_conv1_block2_conv1_kernelblock2_conv1_block2_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_13382&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0 block2_conv2_block2_conv2_kernelblock2_conv2_block2_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_13612&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_11982
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0 block3_conv1_block3_conv1_kernelblock3_conv1_block3_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_13902&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0 block3_conv2_block3_conv2_kernelblock3_conv2_block3_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_14132&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0 block3_conv3_block3_conv3_kernelblock3_conv3_block3_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_14362&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_12152
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0 block4_conv1_block4_conv1_kernelblock4_conv1_block4_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_14652&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0 block4_conv2_block4_conv2_kernelblock4_conv2_block4_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_14882&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0 block4_conv3_block4_conv3_kernelblock4_conv3_block4_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_15112&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_12322
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0 block5_conv1_block5_conv1_kernelblock5_conv1_block5_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_15402&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0 block5_conv2_block5_conv2_kernelblock5_conv2_block5_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_15632&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0 block5_conv3_block5_conv3_kernelblock5_conv3_block5_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_15862&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_12492
block5_pool/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_12682*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?&
?
D__inference_sequential_layer_call_and_return_conditional_losses_3077

inputs,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias0
,dense_3_matmul_readvariableop_dense_3_kernel/
+dense_3_biasadd_readvariableop_dense_3_bias
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu
dropout/IdentityIdentitydense_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAdd?
dense_3/MatMul/ReadVariableOpReadVariableOp,dense_3_matmul_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/BiasAdd:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp+dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Softmax?
IdentityIdentitydense_3/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_3400

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_18892
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv3_layer_call_and_return_conditional_losses_3330

inputs-
)conv2d_readvariableop_block5_conv3_kernel,
(biasadd_readvariableop_block5_conv3_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block5_conv3_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block5_conv3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_block2_conv1_layer_call_and_return_conditional_losses_3150

inputs-
)conv2d_readvariableop_block2_conv1_kernel,
(biasadd_readvariableop_block2_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????pp@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?/
?
D__inference_sequential_layer_call_and_return_conditional_losses_3045

inputs,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias0
,dense_3_matmul_readvariableop_dense_3_kernel/
+dense_3_biasadd_readvariableop_dense_3_bias
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMuldense_1/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mulx
dropout/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mul_1?
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAdd?
dense_3/MatMul/ReadVariableOpReadVariableOp,dense_3_matmul_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/BiasAdd:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp+dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SoftmaxSoftmaxdense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Softmax?
IdentityIdentitydense_3/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block3_conv2_layer_call_and_return_conditional_losses_3204

inputs-
)conv2d_readvariableop_block3_conv2_kernel,
(biasadd_readvariableop_block3_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????88?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?`
?
?__inference_vgg16_layer_call_and_return_conditional_losses_1660
input_1$
 block1_conv1_block1_conv1_kernel"
block1_conv1_block1_conv1_bias$
 block1_conv2_block1_conv2_kernel"
block1_conv2_block1_conv2_bias$
 block2_conv1_block2_conv1_kernel"
block2_conv1_block2_conv1_bias$
 block2_conv2_block2_conv2_kernel"
block2_conv2_block2_conv2_bias$
 block3_conv1_block3_conv1_kernel"
block3_conv1_block3_conv1_bias$
 block3_conv2_block3_conv2_kernel"
block3_conv2_block3_conv2_bias$
 block3_conv3_block3_conv3_kernel"
block3_conv3_block3_conv3_bias$
 block4_conv1_block4_conv1_kernel"
block4_conv1_block4_conv1_bias$
 block4_conv2_block4_conv2_kernel"
block4_conv2_block4_conv2_bias$
 block4_conv3_block4_conv3_kernel"
block4_conv3_block4_conv3_bias$
 block5_conv1_block5_conv1_kernel"
block5_conv1_block5_conv1_bias$
 block5_conv2_block5_conv2_kernel"
block5_conv2_block5_conv2_bias$
 block5_conv3_block5_conv3_kernel"
block5_conv3_block5_conv3_bias
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1 block1_conv1_block1_conv1_kernelblock1_conv1_block1_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_12862&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0 block1_conv2_block1_conv2_kernelblock1_conv2_block1_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_13092&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_11812
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0 block2_conv1_block2_conv1_kernelblock2_conv1_block2_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_13382&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0 block2_conv2_block2_conv2_kernelblock2_conv2_block2_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_13612&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_11982
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0 block3_conv1_block3_conv1_kernelblock3_conv1_block3_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_13902&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0 block3_conv2_block3_conv2_kernelblock3_conv2_block3_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_14132&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0 block3_conv3_block3_conv3_kernelblock3_conv3_block3_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_14362&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_12152
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0 block4_conv1_block4_conv1_kernelblock4_conv1_block4_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_14652&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0 block4_conv2_block4_conv2_kernelblock4_conv2_block4_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_14882&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0 block4_conv3_block4_conv3_kernelblock4_conv3_block4_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_15112&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_12322
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0 block5_conv1_block5_conv1_kernelblock5_conv1_block5_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_15402&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0 block5_conv2_block5_conv2_kernelblock5_conv2_block5_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_15632&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0 block5_conv3_block5_conv3_kernelblock5_conv3_block5_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_15862&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_12492
block5_pool/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_12682*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1948
dense_input
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
dense_3_dense_3_kernel
dense_3_dense_3_bias
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_18362
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_18592!
dense_1/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_18842!
dropout/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_19122!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_19352!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_1224

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_dense_3_layer_call_fn_3435

inputs
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_kerneldense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_19352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_3385

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_block4_conv3_layer_call_fn_3283

inputs
block4_conv3_kernel
block4_conv3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock4_conv3_kernelblock4_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_15112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
?__inference_vgg16_layer_call_and_return_conditional_losses_2944

inputs:
6block1_conv1_conv2d_readvariableop_block1_conv1_kernel9
5block1_conv1_biasadd_readvariableop_block1_conv1_bias:
6block1_conv2_conv2d_readvariableop_block1_conv2_kernel9
5block1_conv2_biasadd_readvariableop_block1_conv2_bias:
6block2_conv1_conv2d_readvariableop_block2_conv1_kernel9
5block2_conv1_biasadd_readvariableop_block2_conv1_bias:
6block2_conv2_conv2d_readvariableop_block2_conv2_kernel9
5block2_conv2_biasadd_readvariableop_block2_conv2_bias:
6block3_conv1_conv2d_readvariableop_block3_conv1_kernel9
5block3_conv1_biasadd_readvariableop_block3_conv1_bias:
6block3_conv2_conv2d_readvariableop_block3_conv2_kernel9
5block3_conv2_biasadd_readvariableop_block3_conv2_bias:
6block3_conv3_conv2d_readvariableop_block3_conv3_kernel9
5block3_conv3_biasadd_readvariableop_block3_conv3_bias:
6block4_conv1_conv2d_readvariableop_block4_conv1_kernel9
5block4_conv1_biasadd_readvariableop_block4_conv1_bias:
6block4_conv2_conv2d_readvariableop_block4_conv2_kernel9
5block4_conv2_biasadd_readvariableop_block4_conv2_bias:
6block4_conv3_conv2d_readvariableop_block4_conv3_kernel9
5block4_conv3_biasadd_readvariableop_block4_conv3_bias:
6block5_conv1_conv2d_readvariableop_block5_conv1_kernel9
5block5_conv1_biasadd_readvariableop_block5_conv1_bias:
6block5_conv2_conv2d_readvariableop_block5_conv2_kernel9
5block5_conv2_biasadd_readvariableop_block5_conv2_bias:
6block5_conv3_conv2d_readvariableop_block5_conv3_kernel9
5block5_conv3_biasadd_readvariableop_block5_conv3_bias
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp6block1_conv1_conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp5block1_conv1_biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp6block1_conv2_conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp5block1_conv2_biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp6block2_conv1_conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp5block2_conv1_biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv1/Relu?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp6block2_conv2_conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp5block2_conv2_biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp6block3_conv1_conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp5block3_conv1_biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv1/Relu?
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp6block3_conv2_conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv2/Conv2D?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp5block3_conv2_biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv2/BiasAdd?
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv2/Relu?
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp6block3_conv3_conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv3/Conv2D?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp5block3_conv3_biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv3/BiasAdd?
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv3/Relu?
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp6block4_conv1_conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv1/Conv2D?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp5block4_conv1_biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv1/BiasAdd?
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv1/Relu?
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp6block4_conv2_conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv2/Conv2D?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp5block4_conv2_biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv2/BiasAdd?
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv2/Relu?
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp6block4_conv3_conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv3/Conv2D?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp5block4_conv3_biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv3/BiasAdd?
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv3/Relu?
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp6block5_conv1_conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv1/Conv2D?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp5block5_conv1_biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv1/BiasAdd?
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv1/Relu?
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp6block5_conv2_conv2d_readvariableop_block5_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block5_conv2/Conv2D/ReadVariableOp?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv2/Conv2D?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp5block5_conv2_biasadd_readvariableop_block5_conv2_bias*
_output_shapes	
:?*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv2/BiasAdd?
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv2/Relu?
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp6block5_conv3_conv2d_readvariableop_block5_conv3_kernel*(
_output_shapes
:??*
dtype02$
"block5_conv3/Conv2D/ReadVariableOp?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv3/Conv2D?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp5block5_conv3_biasadd_readvariableop_block5_conv3_bias*
_output_shapes	
:?*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv3/BiasAdd?
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv3/Relu?
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeanblock5_pool/MaxPool:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
global_average_pooling2d/Mean?
IdentityIdentity&global_average_pooling2d/Mean:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_2662

inputs@
<vgg16_block1_conv1_conv2d_readvariableop_block1_conv1_kernel?
;vgg16_block1_conv1_biasadd_readvariableop_block1_conv1_bias@
<vgg16_block1_conv2_conv2d_readvariableop_block1_conv2_kernel?
;vgg16_block1_conv2_biasadd_readvariableop_block1_conv2_bias@
<vgg16_block2_conv1_conv2d_readvariableop_block2_conv1_kernel?
;vgg16_block2_conv1_biasadd_readvariableop_block2_conv1_bias@
<vgg16_block2_conv2_conv2d_readvariableop_block2_conv2_kernel?
;vgg16_block2_conv2_biasadd_readvariableop_block2_conv2_bias@
<vgg16_block3_conv1_conv2d_readvariableop_block3_conv1_kernel?
;vgg16_block3_conv1_biasadd_readvariableop_block3_conv1_bias@
<vgg16_block3_conv2_conv2d_readvariableop_block3_conv2_kernel?
;vgg16_block3_conv2_biasadd_readvariableop_block3_conv2_bias@
<vgg16_block3_conv3_conv2d_readvariableop_block3_conv3_kernel?
;vgg16_block3_conv3_biasadd_readvariableop_block3_conv3_bias@
<vgg16_block4_conv1_conv2d_readvariableop_block4_conv1_kernel?
;vgg16_block4_conv1_biasadd_readvariableop_block4_conv1_bias@
<vgg16_block4_conv2_conv2d_readvariableop_block4_conv2_kernel?
;vgg16_block4_conv2_biasadd_readvariableop_block4_conv2_bias@
<vgg16_block4_conv3_conv2d_readvariableop_block4_conv3_kernel?
;vgg16_block4_conv3_biasadd_readvariableop_block4_conv3_bias@
<vgg16_block5_conv1_conv2d_readvariableop_block5_conv1_kernel?
;vgg16_block5_conv1_biasadd_readvariableop_block5_conv1_bias@
<vgg16_block5_conv2_conv2d_readvariableop_block5_conv2_kernel?
;vgg16_block5_conv2_biasadd_readvariableop_block5_conv2_bias@
<vgg16_block5_conv3_conv2d_readvariableop_block5_conv3_kernel?
;vgg16_block5_conv3_biasadd_readvariableop_block5_conv3_bias7
3sequential_dense_matmul_readvariableop_dense_kernel6
2sequential_dense_biasadd_readvariableop_dense_bias;
7sequential_dense_1_matmul_readvariableop_dense_1_kernel:
6sequential_dense_1_biasadd_readvariableop_dense_1_bias;
7sequential_dense_2_matmul_readvariableop_dense_2_kernel:
6sequential_dense_2_biasadd_readvariableop_dense_2_bias;
7sequential_dense_3_matmul_readvariableop_dense_3_kernel:
6sequential_dense_3_biasadd_readvariableop_dense_3_bias
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?)sequential/dense_3/BiasAdd/ReadVariableOp?(sequential/dense_3/MatMul/ReadVariableOp?)vgg16/block1_conv1/BiasAdd/ReadVariableOp?(vgg16/block1_conv1/Conv2D/ReadVariableOp?)vgg16/block1_conv2/BiasAdd/ReadVariableOp?(vgg16/block1_conv2/Conv2D/ReadVariableOp?)vgg16/block2_conv1/BiasAdd/ReadVariableOp?(vgg16/block2_conv1/Conv2D/ReadVariableOp?)vgg16/block2_conv2/BiasAdd/ReadVariableOp?(vgg16/block2_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv1/BiasAdd/ReadVariableOp?(vgg16/block3_conv1/Conv2D/ReadVariableOp?)vgg16/block3_conv2/BiasAdd/ReadVariableOp?(vgg16/block3_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv3/BiasAdd/ReadVariableOp?(vgg16/block3_conv3/Conv2D/ReadVariableOp?)vgg16/block4_conv1/BiasAdd/ReadVariableOp?(vgg16/block4_conv1/Conv2D/ReadVariableOp?)vgg16/block4_conv2/BiasAdd/ReadVariableOp?(vgg16/block4_conv2/Conv2D/ReadVariableOp?)vgg16/block4_conv3/BiasAdd/ReadVariableOp?(vgg16/block4_conv3/Conv2D/ReadVariableOp?)vgg16/block5_conv1/BiasAdd/ReadVariableOp?(vgg16/block5_conv1/Conv2D/ReadVariableOp?)vgg16/block5_conv2/BiasAdd/ReadVariableOp?(vgg16/block5_conv2/Conv2D/ReadVariableOp?)vgg16/block5_conv3/BiasAdd/ReadVariableOp?(vgg16/block5_conv3/Conv2D/ReadVariableOp?
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_block1_conv1_conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02*
(vgg16/block1_conv1/Conv2D/ReadVariableOp?
vgg16/block1_conv1/Conv2DConv2Dinputs0vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv1/Conv2D?
)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block1_conv1_biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv1/BiasAdd/ReadVariableOp?
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/BiasAdd?
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/Relu?
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_block1_conv2_conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02*
(vgg16/block1_conv2/Conv2D/ReadVariableOp?
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv2/Conv2D?
)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block1_conv2_biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv2/BiasAdd/ReadVariableOp?
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/BiasAdd?
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/Relu?
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
vgg16/block1_pool/MaxPool?
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_block2_conv1_conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02*
(vgg16/block2_conv1/Conv2D/ReadVariableOp?
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
vgg16/block2_conv1/Conv2D?
)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block2_conv1_biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv1/BiasAdd/ReadVariableOp?
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv1/BiasAdd?
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv1/Relu?
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_block2_conv2_conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block2_conv2/Conv2D/ReadVariableOp?
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
vgg16/block2_conv2/Conv2D?
)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block2_conv2_biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv2/BiasAdd/ReadVariableOp?
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv2/BiasAdd?
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv2/Relu?
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
vgg16/block2_pool/MaxPool?
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_block3_conv1_conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv1/Conv2D/ReadVariableOp?
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv1/Conv2D?
)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block3_conv1_biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv1/BiasAdd/ReadVariableOp?
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv1/BiasAdd?
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv1/Relu?
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_block3_conv2_conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv2/Conv2D/ReadVariableOp?
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv2/Conv2D?
)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block3_conv2_biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv2/BiasAdd/ReadVariableOp?
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv2/BiasAdd?
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv2/Relu?
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp<vgg16_block3_conv3_conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv3/Conv2D/ReadVariableOp?
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv3/Conv2D?
)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block3_conv3_biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv3/BiasAdd/ReadVariableOp?
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv3/BiasAdd?
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv3/Relu?
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block3_pool/MaxPool?
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_block4_conv1_conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv1/Conv2D/ReadVariableOp?
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv1/Conv2D?
)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block4_conv1_biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv1/BiasAdd/ReadVariableOp?
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv1/BiasAdd?
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv1/Relu?
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_block4_conv2_conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv2/Conv2D/ReadVariableOp?
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv2/Conv2D?
)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block4_conv2_biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv2/BiasAdd/ReadVariableOp?
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv2/BiasAdd?
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv2/Relu?
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp<vgg16_block4_conv3_conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv3/Conv2D/ReadVariableOp?
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv3/Conv2D?
)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block4_conv3_biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv3/BiasAdd/ReadVariableOp?
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv3/BiasAdd?
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv3/Relu?
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block4_pool/MaxPool?
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_block5_conv1_conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv1/Conv2D/ReadVariableOp?
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv1/Conv2D?
)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block5_conv1_biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv1/BiasAdd/ReadVariableOp?
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/BiasAdd?
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/Relu?
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_block5_conv2_conv2d_readvariableop_block5_conv2_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv2/Conv2D/ReadVariableOp?
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv2/Conv2D?
)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block5_conv2_biasadd_readvariableop_block5_conv2_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv2/BiasAdd/ReadVariableOp?
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/BiasAdd?
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/Relu?
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp<vgg16_block5_conv3_conv2d_readvariableop_block5_conv3_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv3/Conv2D/ReadVariableOp?
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv3/Conv2D?
)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block5_conv3_biasadd_readvariableop_block5_conv3_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv3/BiasAdd/ReadVariableOp?
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/BiasAdd?
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/Relu?
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block5_pool/MaxPool?
5vgg16/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5vgg16/global_average_pooling2d/Mean/reduction_indices?
#vgg16/global_average_pooling2d/MeanMean"vgg16/block5_pool/MaxPool:output:0>vgg16/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2%
#vgg16/global_average_pooling2d/Mean?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp3sequential_dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul,vgg16/global_average_pooling2d/Mean:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7sequential_dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
sequential/dropout/IdentityIdentity%sequential/dense_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2
sequential/dropout/Identity?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp7sequential_dense_2_matmul_readvariableop_dense_2_kernel* 
_output_shapes
:
??*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes	
:?*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_2/BiasAdd?
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp7sequential_dense_3_matmul_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype02*
(sequential/dense_3/MatMul/ReadVariableOp?
sequential/dense_3/MatMulMatMul#sequential/dense_2/BiasAdd:output:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_3/MatMul?
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes
:*
dtype02+
)sequential/dense_3/BiasAdd/ReadVariableOp?
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_3/BiasAdd?
sequential/dense_3/SoftmaxSoftmax#sequential/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_3/Softmax?
IdentityIdentity$sequential/dense_3/Softmax:softmax:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp2V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv2_layer_call_and_return_conditional_losses_3312

inputs-
)conv2d_readvariableop_block5_conv2_kernel,
(biasadd_readvariableop_block5_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block5_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block5_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_2275
input_2
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias
block5_conv2_kernel
block5_conv2_bias
block5_conv3_kernel
block5_conv3_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biasblock5_conv2_kernelblock5_conv2_biasblock5_conv3_kernelblock5_conv3_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_22382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
+__inference_block1_conv2_layer_call_fn_3139

inputs
block1_conv2_kernel
block1_conv2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv2_kernelblock1_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_13092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
F__inference_block4_conv3_layer_call_and_return_conditional_losses_3276

inputs-
)conv2d_readvariableop_block4_conv3_kernel,
(biasadd_readvariableop_block4_conv3_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_2740

inputs
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias
block5_conv2_kernel
block5_conv2_bias
block5_conv3_kernel
block5_conv3_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biasblock5_conv2_kernelblock5_conv2_biasblock5_conv3_kernelblock5_conv3_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_23172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_block5_conv2_layer_call_fn_3319

inputs
block5_conv2_kernel
block5_conv2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock5_conv2_kernelblock5_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_15632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv3_layer_call_and_return_conditional_losses_1511

inputs-
)conv2d_readvariableop_block4_conv3_kernel,
(biasadd_readvariableop_block4_conv3_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_2354
input_2
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias
block5_conv2_kernel
block5_conv2_bias
block5_conv3_kernel
block5_conv3_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biasblock5_conv2_kernelblock5_conv2_biasblock5_conv3_kernelblock5_conv3_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_23172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
??
?
?__inference_model_layer_call_and_return_conditional_losses_2532

inputs@
<vgg16_block1_conv1_conv2d_readvariableop_block1_conv1_kernel?
;vgg16_block1_conv1_biasadd_readvariableop_block1_conv1_bias@
<vgg16_block1_conv2_conv2d_readvariableop_block1_conv2_kernel?
;vgg16_block1_conv2_biasadd_readvariableop_block1_conv2_bias@
<vgg16_block2_conv1_conv2d_readvariableop_block2_conv1_kernel?
;vgg16_block2_conv1_biasadd_readvariableop_block2_conv1_bias@
<vgg16_block2_conv2_conv2d_readvariableop_block2_conv2_kernel?
;vgg16_block2_conv2_biasadd_readvariableop_block2_conv2_bias@
<vgg16_block3_conv1_conv2d_readvariableop_block3_conv1_kernel?
;vgg16_block3_conv1_biasadd_readvariableop_block3_conv1_bias@
<vgg16_block3_conv2_conv2d_readvariableop_block3_conv2_kernel?
;vgg16_block3_conv2_biasadd_readvariableop_block3_conv2_bias@
<vgg16_block3_conv3_conv2d_readvariableop_block3_conv3_kernel?
;vgg16_block3_conv3_biasadd_readvariableop_block3_conv3_bias@
<vgg16_block4_conv1_conv2d_readvariableop_block4_conv1_kernel?
;vgg16_block4_conv1_biasadd_readvariableop_block4_conv1_bias@
<vgg16_block4_conv2_conv2d_readvariableop_block4_conv2_kernel?
;vgg16_block4_conv2_biasadd_readvariableop_block4_conv2_bias@
<vgg16_block4_conv3_conv2d_readvariableop_block4_conv3_kernel?
;vgg16_block4_conv3_biasadd_readvariableop_block4_conv3_bias@
<vgg16_block5_conv1_conv2d_readvariableop_block5_conv1_kernel?
;vgg16_block5_conv1_biasadd_readvariableop_block5_conv1_bias@
<vgg16_block5_conv2_conv2d_readvariableop_block5_conv2_kernel?
;vgg16_block5_conv2_biasadd_readvariableop_block5_conv2_bias@
<vgg16_block5_conv3_conv2d_readvariableop_block5_conv3_kernel?
;vgg16_block5_conv3_biasadd_readvariableop_block5_conv3_bias7
3sequential_dense_matmul_readvariableop_dense_kernel6
2sequential_dense_biasadd_readvariableop_dense_bias;
7sequential_dense_1_matmul_readvariableop_dense_1_kernel:
6sequential_dense_1_biasadd_readvariableop_dense_1_bias;
7sequential_dense_2_matmul_readvariableop_dense_2_kernel:
6sequential_dense_2_biasadd_readvariableop_dense_2_bias;
7sequential_dense_3_matmul_readvariableop_dense_3_kernel:
6sequential_dense_3_biasadd_readvariableop_dense_3_bias
identity??'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?)sequential/dense_3/BiasAdd/ReadVariableOp?(sequential/dense_3/MatMul/ReadVariableOp?)vgg16/block1_conv1/BiasAdd/ReadVariableOp?(vgg16/block1_conv1/Conv2D/ReadVariableOp?)vgg16/block1_conv2/BiasAdd/ReadVariableOp?(vgg16/block1_conv2/Conv2D/ReadVariableOp?)vgg16/block2_conv1/BiasAdd/ReadVariableOp?(vgg16/block2_conv1/Conv2D/ReadVariableOp?)vgg16/block2_conv2/BiasAdd/ReadVariableOp?(vgg16/block2_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv1/BiasAdd/ReadVariableOp?(vgg16/block3_conv1/Conv2D/ReadVariableOp?)vgg16/block3_conv2/BiasAdd/ReadVariableOp?(vgg16/block3_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv3/BiasAdd/ReadVariableOp?(vgg16/block3_conv3/Conv2D/ReadVariableOp?)vgg16/block4_conv1/BiasAdd/ReadVariableOp?(vgg16/block4_conv1/Conv2D/ReadVariableOp?)vgg16/block4_conv2/BiasAdd/ReadVariableOp?(vgg16/block4_conv2/Conv2D/ReadVariableOp?)vgg16/block4_conv3/BiasAdd/ReadVariableOp?(vgg16/block4_conv3/Conv2D/ReadVariableOp?)vgg16/block5_conv1/BiasAdd/ReadVariableOp?(vgg16/block5_conv1/Conv2D/ReadVariableOp?)vgg16/block5_conv2/BiasAdd/ReadVariableOp?(vgg16/block5_conv2/Conv2D/ReadVariableOp?)vgg16/block5_conv3/BiasAdd/ReadVariableOp?(vgg16/block5_conv3/Conv2D/ReadVariableOp?
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_block1_conv1_conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02*
(vgg16/block1_conv1/Conv2D/ReadVariableOp?
vgg16/block1_conv1/Conv2DConv2Dinputs0vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv1/Conv2D?
)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block1_conv1_biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv1/BiasAdd/ReadVariableOp?
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/BiasAdd?
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv1/Relu?
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_block1_conv2_conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02*
(vgg16/block1_conv2/Conv2D/ReadVariableOp?
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
vgg16/block1_conv2/Conv2D?
)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block1_conv2_biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02+
)vgg16/block1_conv2/BiasAdd/ReadVariableOp?
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/BiasAdd?
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
vgg16/block1_conv2/Relu?
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
vgg16/block1_pool/MaxPool?
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_block2_conv1_conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02*
(vgg16/block2_conv1/Conv2D/ReadVariableOp?
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
vgg16/block2_conv1/Conv2D?
)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block2_conv1_biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv1/BiasAdd/ReadVariableOp?
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv1/BiasAdd?
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv1/Relu?
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_block2_conv2_conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block2_conv2/Conv2D/ReadVariableOp?
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
vgg16/block2_conv2/Conv2D?
)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block2_conv2_biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block2_conv2/BiasAdd/ReadVariableOp?
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv2/BiasAdd?
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
vgg16/block2_conv2/Relu?
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
vgg16/block2_pool/MaxPool?
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_block3_conv1_conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv1/Conv2D/ReadVariableOp?
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv1/Conv2D?
)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block3_conv1_biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv1/BiasAdd/ReadVariableOp?
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv1/BiasAdd?
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv1/Relu?
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_block3_conv2_conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv2/Conv2D/ReadVariableOp?
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv2/Conv2D?
)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block3_conv2_biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv2/BiasAdd/ReadVariableOp?
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv2/BiasAdd?
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv2/Relu?
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp<vgg16_block3_conv3_conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block3_conv3/Conv2D/ReadVariableOp?
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
vgg16/block3_conv3/Conv2D?
)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block3_conv3_biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block3_conv3/BiasAdd/ReadVariableOp?
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv3/BiasAdd?
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
vgg16/block3_conv3/Relu?
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block3_pool/MaxPool?
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_block4_conv1_conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv1/Conv2D/ReadVariableOp?
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv1/Conv2D?
)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block4_conv1_biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv1/BiasAdd/ReadVariableOp?
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv1/BiasAdd?
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv1/Relu?
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_block4_conv2_conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv2/Conv2D/ReadVariableOp?
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv2/Conv2D?
)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block4_conv2_biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv2/BiasAdd/ReadVariableOp?
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv2/BiasAdd?
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv2/Relu?
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp<vgg16_block4_conv3_conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block4_conv3/Conv2D/ReadVariableOp?
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block4_conv3/Conv2D?
)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block4_conv3_biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block4_conv3/BiasAdd/ReadVariableOp?
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv3/BiasAdd?
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block4_conv3/Relu?
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block4_pool/MaxPool?
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_block5_conv1_conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv1/Conv2D/ReadVariableOp?
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv1/Conv2D?
)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block5_conv1_biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv1/BiasAdd/ReadVariableOp?
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/BiasAdd?
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv1/Relu?
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_block5_conv2_conv2d_readvariableop_block5_conv2_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv2/Conv2D/ReadVariableOp?
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv2/Conv2D?
)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block5_conv2_biasadd_readvariableop_block5_conv2_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv2/BiasAdd/ReadVariableOp?
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/BiasAdd?
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv2/Relu?
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp<vgg16_block5_conv3_conv2d_readvariableop_block5_conv3_kernel*(
_output_shapes
:??*
dtype02*
(vgg16/block5_conv3/Conv2D/ReadVariableOp?
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
vgg16/block5_conv3/Conv2D?
)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp;vgg16_block5_conv3_biasadd_readvariableop_block5_conv3_bias*
_output_shapes	
:?*
dtype02+
)vgg16/block5_conv3/BiasAdd/ReadVariableOp?
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/BiasAdd?
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
vgg16/block5_conv3/Relu?
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
vgg16/block5_pool/MaxPool?
5vgg16/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5vgg16/global_average_pooling2d/Mean/reduction_indices?
#vgg16/global_average_pooling2d/MeanMean"vgg16/block5_pool/MaxPool:output:0>vgg16/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2%
#vgg16/global_average_pooling2d/Mean?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp3sequential_dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
??*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul,vgg16/global_average_pooling2d/Mean:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense/Relu?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp7sequential_dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
??*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:?*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/dense_1/Relu?
 sequential/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 sequential/dropout/dropout/Const?
sequential/dropout/dropout/MulMul%sequential/dense_1/Relu:activations:0)sequential/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential/dropout/dropout/Mul?
 sequential/dropout/dropout/ShapeShape%sequential/dense_1/Relu:activations:0*
T0*
_output_shapes
:2"
 sequential/dropout/dropout/Shape?
7sequential/dropout/dropout/random_uniform/RandomUniformRandomUniform)sequential/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype029
7sequential/dropout/dropout/random_uniform/RandomUniform?
)sequential/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)sequential/dropout/dropout/GreaterEqual/y?
'sequential/dropout/dropout/GreaterEqualGreaterEqual@sequential/dropout/dropout/random_uniform/RandomUniform:output:02sequential/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'sequential/dropout/dropout/GreaterEqual?
sequential/dropout/dropout/CastCast+sequential/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
sequential/dropout/dropout/Cast?
 sequential/dropout/dropout/Mul_1Mul"sequential/dropout/dropout/Mul:z:0#sequential/dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 sequential/dropout/dropout/Mul_1?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp7sequential_dense_2_matmul_readvariableop_dense_2_kernel* 
_output_shapes
:
??*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMul$sequential/dropout/dropout/Mul_1:z:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes	
:?*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/dense_2/BiasAdd?
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp7sequential_dense_3_matmul_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype02*
(sequential/dense_3/MatMul/ReadVariableOp?
sequential/dense_3/MatMulMatMul#sequential/dense_2/BiasAdd:output:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_3/MatMul?
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes
:*
dtype02+
)sequential/dense_3/BiasAdd/ReadVariableOp?
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_3/BiasAdd?
sequential/dense_3/SoftmaxSoftmax#sequential/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_3/Softmax?
IdentityIdentity$sequential/dense_3/Softmax:softmax:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp2V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_1215

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_1836

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1259

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?`
?
?__inference_vgg16_layer_call_and_return_conditional_losses_1792

inputs$
 block1_conv1_block1_conv1_kernel"
block1_conv1_block1_conv1_bias$
 block1_conv2_block1_conv2_kernel"
block1_conv2_block1_conv2_bias$
 block2_conv1_block2_conv1_kernel"
block2_conv1_block2_conv1_bias$
 block2_conv2_block2_conv2_kernel"
block2_conv2_block2_conv2_bias$
 block3_conv1_block3_conv1_kernel"
block3_conv1_block3_conv1_bias$
 block3_conv2_block3_conv2_kernel"
block3_conv2_block3_conv2_bias$
 block3_conv3_block3_conv3_kernel"
block3_conv3_block3_conv3_bias$
 block4_conv1_block4_conv1_kernel"
block4_conv1_block4_conv1_bias$
 block4_conv2_block4_conv2_kernel"
block4_conv2_block4_conv2_bias$
 block4_conv3_block4_conv3_kernel"
block4_conv3_block4_conv3_bias$
 block5_conv1_block5_conv1_kernel"
block5_conv1_block5_conv1_bias$
 block5_conv2_block5_conv2_kernel"
block5_conv2_block5_conv2_bias$
 block5_conv3_block5_conv3_kernel"
block5_conv3_block5_conv3_bias
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputs block1_conv1_block1_conv1_kernelblock1_conv1_block1_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_12862&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0 block1_conv2_block1_conv2_kernelblock1_conv2_block1_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_13092&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_11812
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0 block2_conv1_block2_conv1_kernelblock2_conv1_block2_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_13382&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0 block2_conv2_block2_conv2_kernelblock2_conv2_block2_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_13612&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_11982
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0 block3_conv1_block3_conv1_kernelblock3_conv1_block3_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_13902&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0 block3_conv2_block3_conv2_kernelblock3_conv2_block3_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_14132&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0 block3_conv3_block3_conv3_kernelblock3_conv3_block3_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_14362&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_12152
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0 block4_conv1_block4_conv1_kernelblock4_conv1_block4_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_14652&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0 block4_conv2_block4_conv2_kernelblock4_conv2_block4_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_14882&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0 block4_conv3_block4_conv3_kernelblock4_conv3_block4_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_15112&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_12322
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0 block5_conv1_block5_conv1_kernelblock5_conv1_block5_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_15402&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0 block5_conv2_block5_conv2_kernelblock5_conv2_block5_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_15632&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0 block5_conv3_block5_conv3_kernelblock5_conv3_block5_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_15862&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_12492
block5_pool/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_12682*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
A__inference_dense_3_layer_call_and_return_conditional_losses_1935

inputs(
$matmul_readvariableop_dense_3_kernel'
#biasadd_readvariableop_dense_3_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_3_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_1232

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_3390

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_dense_2_layer_call_fn_3417

inputs
dense_2_kernel
dense_2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_19122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_1996
dense_input
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_19852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?
?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_1309

inputs-
)conv2d_readvariableop_block1_conv2_kernel,
(biasadd_readvariableop_block1_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
$__inference_vgg16_layer_call_fn_1741
input_1
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias
block5_conv2_kernel
block5_conv2_bias
block5_conv3_kernel
block5_conv3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biasblock5_conv2_kernelblock5_conv2_biasblock5_conv3_kernelblock5_conv3_bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_17122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
+__inference_block3_conv2_layer_call_fn_3211

inputs
block3_conv2_kernel
block3_conv2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock3_conv2_kernelblock3_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_14132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????88?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_3366

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_block5_pool_layer_call_fn_1252

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_12492
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block2_conv2_layer_call_and_return_conditional_losses_3168

inputs-
)conv2d_readvariableop_block2_conv2_kernel,
(biasadd_readvariableop_block2_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????pp?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?
?	
?__inference_model_layer_call_and_return_conditional_losses_2195
input_2
vgg16_block1_conv1_kernel
vgg16_block1_conv1_bias
vgg16_block1_conv2_kernel
vgg16_block1_conv2_bias
vgg16_block2_conv1_kernel
vgg16_block2_conv1_bias
vgg16_block2_conv2_kernel
vgg16_block2_conv2_bias
vgg16_block3_conv1_kernel
vgg16_block3_conv1_bias
vgg16_block3_conv2_kernel
vgg16_block3_conv2_bias
vgg16_block3_conv3_kernel
vgg16_block3_conv3_bias
vgg16_block4_conv1_kernel
vgg16_block4_conv1_bias
vgg16_block4_conv2_kernel
vgg16_block4_conv2_bias
vgg16_block4_conv3_kernel
vgg16_block4_conv3_bias
vgg16_block5_conv1_kernel
vgg16_block5_conv1_bias
vgg16_block5_conv2_kernel
vgg16_block5_conv2_bias
vgg16_block5_conv3_kernel
vgg16_block5_conv3_bias
sequential_dense_kernel
sequential_dense_bias
sequential_dense_1_kernel
sequential_dense_1_bias
sequential_dense_2_kernel
sequential_dense_2_bias
sequential_dense_3_kernel
sequential_dense_3_bias
identity??"sequential/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?
vgg16/StatefulPartitionedCallStatefulPartitionedCallinput_2vgg16_block1_conv1_kernelvgg16_block1_conv1_biasvgg16_block1_conv2_kernelvgg16_block1_conv2_biasvgg16_block2_conv1_kernelvgg16_block2_conv1_biasvgg16_block2_conv2_kernelvgg16_block2_conv2_biasvgg16_block3_conv1_kernelvgg16_block3_conv1_biasvgg16_block3_conv2_kernelvgg16_block3_conv2_biasvgg16_block3_conv3_kernelvgg16_block3_conv3_biasvgg16_block4_conv1_kernelvgg16_block4_conv1_biasvgg16_block4_conv2_kernelvgg16_block4_conv2_biasvgg16_block4_conv3_kernelvgg16_block4_conv3_biasvgg16_block5_conv1_kernelvgg16_block5_conv1_biasvgg16_block5_conv2_kernelvgg16_block5_conv2_biasvgg16_block5_conv3_kernelvgg16_block5_conv3_bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_17922
vgg16/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0sequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_biassequential_dense_3_kernelsequential_dense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_20152$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
F__inference_block5_conv3_layer_call_and_return_conditional_losses_1586

inputs-
)conv2d_readvariableop_block5_conv3_kernel,
(biasadd_readvariableop_block5_conv3_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block5_conv3_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block5_conv3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block2_conv2_layer_call_and_return_conditional_losses_1361

inputs-
)conv2d_readvariableop_block2_conv2_kernel,
(biasadd_readvariableop_block2_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????pp?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?
?
+__inference_block2_conv1_layer_call_fn_3157

inputs
block2_conv1_kernel
block2_conv1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock2_conv1_kernelblock2_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_13382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????pp@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?
?
F__inference_block3_conv2_layer_call_and_return_conditional_losses_1413

inputs-
)conv2d_readvariableop_block3_conv2_kernel,
(biasadd_readvariableop_block3_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????88?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
S
7__inference_global_average_pooling2d_layer_call_fn_1271

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_12682
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
F__inference_block2_conv1_layer_call_and_return_conditional_losses_1338

inputs-
)conv2d_readvariableop_block2_conv1_kernel,
(biasadd_readvariableop_block2_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????pp?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????pp@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?
F
*__inference_block4_pool_layer_call_fn_1235

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_12322
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
? 
__inference__wrapped_model_1167
input_2F
Bmodel_vgg16_block1_conv1_conv2d_readvariableop_block1_conv1_kernelE
Amodel_vgg16_block1_conv1_biasadd_readvariableop_block1_conv1_biasF
Bmodel_vgg16_block1_conv2_conv2d_readvariableop_block1_conv2_kernelE
Amodel_vgg16_block1_conv2_biasadd_readvariableop_block1_conv2_biasF
Bmodel_vgg16_block2_conv1_conv2d_readvariableop_block2_conv1_kernelE
Amodel_vgg16_block2_conv1_biasadd_readvariableop_block2_conv1_biasF
Bmodel_vgg16_block2_conv2_conv2d_readvariableop_block2_conv2_kernelE
Amodel_vgg16_block2_conv2_biasadd_readvariableop_block2_conv2_biasF
Bmodel_vgg16_block3_conv1_conv2d_readvariableop_block3_conv1_kernelE
Amodel_vgg16_block3_conv1_biasadd_readvariableop_block3_conv1_biasF
Bmodel_vgg16_block3_conv2_conv2d_readvariableop_block3_conv2_kernelE
Amodel_vgg16_block3_conv2_biasadd_readvariableop_block3_conv2_biasF
Bmodel_vgg16_block3_conv3_conv2d_readvariableop_block3_conv3_kernelE
Amodel_vgg16_block3_conv3_biasadd_readvariableop_block3_conv3_biasF
Bmodel_vgg16_block4_conv1_conv2d_readvariableop_block4_conv1_kernelE
Amodel_vgg16_block4_conv1_biasadd_readvariableop_block4_conv1_biasF
Bmodel_vgg16_block4_conv2_conv2d_readvariableop_block4_conv2_kernelE
Amodel_vgg16_block4_conv2_biasadd_readvariableop_block4_conv2_biasF
Bmodel_vgg16_block4_conv3_conv2d_readvariableop_block4_conv3_kernelE
Amodel_vgg16_block4_conv3_biasadd_readvariableop_block4_conv3_biasF
Bmodel_vgg16_block5_conv1_conv2d_readvariableop_block5_conv1_kernelE
Amodel_vgg16_block5_conv1_biasadd_readvariableop_block5_conv1_biasF
Bmodel_vgg16_block5_conv2_conv2d_readvariableop_block5_conv2_kernelE
Amodel_vgg16_block5_conv2_biasadd_readvariableop_block5_conv2_biasF
Bmodel_vgg16_block5_conv3_conv2d_readvariableop_block5_conv3_kernelE
Amodel_vgg16_block5_conv3_biasadd_readvariableop_block5_conv3_bias=
9model_sequential_dense_matmul_readvariableop_dense_kernel<
8model_sequential_dense_biasadd_readvariableop_dense_biasA
=model_sequential_dense_1_matmul_readvariableop_dense_1_kernel@
<model_sequential_dense_1_biasadd_readvariableop_dense_1_biasA
=model_sequential_dense_2_matmul_readvariableop_dense_2_kernel@
<model_sequential_dense_2_biasadd_readvariableop_dense_2_biasA
=model_sequential_dense_3_matmul_readvariableop_dense_3_kernel@
<model_sequential_dense_3_biasadd_readvariableop_dense_3_bias
identity??-model/sequential/dense/BiasAdd/ReadVariableOp?,model/sequential/dense/MatMul/ReadVariableOp?/model/sequential/dense_1/BiasAdd/ReadVariableOp?.model/sequential/dense_1/MatMul/ReadVariableOp?/model/sequential/dense_2/BiasAdd/ReadVariableOp?.model/sequential/dense_2/MatMul/ReadVariableOp?/model/sequential/dense_3/BiasAdd/ReadVariableOp?.model/sequential/dense_3/MatMul/ReadVariableOp?/model/vgg16/block1_conv1/BiasAdd/ReadVariableOp?.model/vgg16/block1_conv1/Conv2D/ReadVariableOp?/model/vgg16/block1_conv2/BiasAdd/ReadVariableOp?.model/vgg16/block1_conv2/Conv2D/ReadVariableOp?/model/vgg16/block2_conv1/BiasAdd/ReadVariableOp?.model/vgg16/block2_conv1/Conv2D/ReadVariableOp?/model/vgg16/block2_conv2/BiasAdd/ReadVariableOp?.model/vgg16/block2_conv2/Conv2D/ReadVariableOp?/model/vgg16/block3_conv1/BiasAdd/ReadVariableOp?.model/vgg16/block3_conv1/Conv2D/ReadVariableOp?/model/vgg16/block3_conv2/BiasAdd/ReadVariableOp?.model/vgg16/block3_conv2/Conv2D/ReadVariableOp?/model/vgg16/block3_conv3/BiasAdd/ReadVariableOp?.model/vgg16/block3_conv3/Conv2D/ReadVariableOp?/model/vgg16/block4_conv1/BiasAdd/ReadVariableOp?.model/vgg16/block4_conv1/Conv2D/ReadVariableOp?/model/vgg16/block4_conv2/BiasAdd/ReadVariableOp?.model/vgg16/block4_conv2/Conv2D/ReadVariableOp?/model/vgg16/block4_conv3/BiasAdd/ReadVariableOp?.model/vgg16/block4_conv3/Conv2D/ReadVariableOp?/model/vgg16/block5_conv1/BiasAdd/ReadVariableOp?.model/vgg16/block5_conv1/Conv2D/ReadVariableOp?/model/vgg16/block5_conv2/BiasAdd/ReadVariableOp?.model/vgg16/block5_conv2/Conv2D/ReadVariableOp?/model/vgg16/block5_conv3/BiasAdd/ReadVariableOp?.model/vgg16/block5_conv3/Conv2D/ReadVariableOp?
.model/vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block1_conv1_conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype020
.model/vgg16/block1_conv1/Conv2D/ReadVariableOp?
model/vgg16/block1_conv1/Conv2DConv2Dinput_26model/vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2!
model/vgg16/block1_conv1/Conv2D?
/model/vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block1_conv1_biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype021
/model/vgg16/block1_conv1/BiasAdd/ReadVariableOp?
 model/vgg16/block1_conv1/BiasAddBiasAdd(model/vgg16/block1_conv1/Conv2D:output:07model/vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2"
 model/vgg16/block1_conv1/BiasAdd?
model/vgg16/block1_conv1/ReluRelu)model/vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
model/vgg16/block1_conv1/Relu?
.model/vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block1_conv2_conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype020
.model/vgg16/block1_conv2/Conv2D/ReadVariableOp?
model/vgg16/block1_conv2/Conv2DConv2D+model/vgg16/block1_conv1/Relu:activations:06model/vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2!
model/vgg16/block1_conv2/Conv2D?
/model/vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block1_conv2_biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype021
/model/vgg16/block1_conv2/BiasAdd/ReadVariableOp?
 model/vgg16/block1_conv2/BiasAddBiasAdd(model/vgg16/block1_conv2/Conv2D:output:07model/vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2"
 model/vgg16/block1_conv2/BiasAdd?
model/vgg16/block1_conv2/ReluRelu)model/vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
model/vgg16/block1_conv2/Relu?
model/vgg16/block1_pool/MaxPoolMaxPool+model/vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2!
model/vgg16/block1_pool/MaxPool?
.model/vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block2_conv1_conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype020
.model/vgg16/block2_conv1/Conv2D/ReadVariableOp?
model/vgg16/block2_conv1/Conv2DConv2D(model/vgg16/block1_pool/MaxPool:output:06model/vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2!
model/vgg16/block2_conv1/Conv2D?
/model/vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block2_conv1_biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block2_conv1/BiasAdd/ReadVariableOp?
 model/vgg16/block2_conv1/BiasAddBiasAdd(model/vgg16/block2_conv1/Conv2D:output:07model/vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2"
 model/vgg16/block2_conv1/BiasAdd?
model/vgg16/block2_conv1/ReluRelu)model/vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
model/vgg16/block2_conv1/Relu?
.model/vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block2_conv2_conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype020
.model/vgg16/block2_conv2/Conv2D/ReadVariableOp?
model/vgg16/block2_conv2/Conv2DConv2D+model/vgg16/block2_conv1/Relu:activations:06model/vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2!
model/vgg16/block2_conv2/Conv2D?
/model/vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block2_conv2_biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block2_conv2/BiasAdd/ReadVariableOp?
 model/vgg16/block2_conv2/BiasAddBiasAdd(model/vgg16/block2_conv2/Conv2D:output:07model/vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2"
 model/vgg16/block2_conv2/BiasAdd?
model/vgg16/block2_conv2/ReluRelu)model/vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
model/vgg16/block2_conv2/Relu?
model/vgg16/block2_pool/MaxPoolMaxPool+model/vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2!
model/vgg16/block2_pool/MaxPool?
.model/vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block3_conv1_conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype020
.model/vgg16/block3_conv1/Conv2D/ReadVariableOp?
model/vgg16/block3_conv1/Conv2DConv2D(model/vgg16/block2_pool/MaxPool:output:06model/vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2!
model/vgg16/block3_conv1/Conv2D?
/model/vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block3_conv1_biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block3_conv1/BiasAdd/ReadVariableOp?
 model/vgg16/block3_conv1/BiasAddBiasAdd(model/vgg16/block3_conv1/Conv2D:output:07model/vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2"
 model/vgg16/block3_conv1/BiasAdd?
model/vgg16/block3_conv1/ReluRelu)model/vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
model/vgg16/block3_conv1/Relu?
.model/vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block3_conv2_conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype020
.model/vgg16/block3_conv2/Conv2D/ReadVariableOp?
model/vgg16/block3_conv2/Conv2DConv2D+model/vgg16/block3_conv1/Relu:activations:06model/vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2!
model/vgg16/block3_conv2/Conv2D?
/model/vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block3_conv2_biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block3_conv2/BiasAdd/ReadVariableOp?
 model/vgg16/block3_conv2/BiasAddBiasAdd(model/vgg16/block3_conv2/Conv2D:output:07model/vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2"
 model/vgg16/block3_conv2/BiasAdd?
model/vgg16/block3_conv2/ReluRelu)model/vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
model/vgg16/block3_conv2/Relu?
.model/vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block3_conv3_conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype020
.model/vgg16/block3_conv3/Conv2D/ReadVariableOp?
model/vgg16/block3_conv3/Conv2DConv2D+model/vgg16/block3_conv2/Relu:activations:06model/vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2!
model/vgg16/block3_conv3/Conv2D?
/model/vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block3_conv3_biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block3_conv3/BiasAdd/ReadVariableOp?
 model/vgg16/block3_conv3/BiasAddBiasAdd(model/vgg16/block3_conv3/Conv2D:output:07model/vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2"
 model/vgg16/block3_conv3/BiasAdd?
model/vgg16/block3_conv3/ReluRelu)model/vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
model/vgg16/block3_conv3/Relu?
model/vgg16/block3_pool/MaxPoolMaxPool+model/vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2!
model/vgg16/block3_pool/MaxPool?
.model/vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block4_conv1_conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype020
.model/vgg16/block4_conv1/Conv2D/ReadVariableOp?
model/vgg16/block4_conv1/Conv2DConv2D(model/vgg16/block3_pool/MaxPool:output:06model/vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2!
model/vgg16/block4_conv1/Conv2D?
/model/vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block4_conv1_biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block4_conv1/BiasAdd/ReadVariableOp?
 model/vgg16/block4_conv1/BiasAddBiasAdd(model/vgg16/block4_conv1/Conv2D:output:07model/vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/vgg16/block4_conv1/BiasAdd?
model/vgg16/block4_conv1/ReluRelu)model/vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/vgg16/block4_conv1/Relu?
.model/vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block4_conv2_conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype020
.model/vgg16/block4_conv2/Conv2D/ReadVariableOp?
model/vgg16/block4_conv2/Conv2DConv2D+model/vgg16/block4_conv1/Relu:activations:06model/vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2!
model/vgg16/block4_conv2/Conv2D?
/model/vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block4_conv2_biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block4_conv2/BiasAdd/ReadVariableOp?
 model/vgg16/block4_conv2/BiasAddBiasAdd(model/vgg16/block4_conv2/Conv2D:output:07model/vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/vgg16/block4_conv2/BiasAdd?
model/vgg16/block4_conv2/ReluRelu)model/vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/vgg16/block4_conv2/Relu?
.model/vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block4_conv3_conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype020
.model/vgg16/block4_conv3/Conv2D/ReadVariableOp?
model/vgg16/block4_conv3/Conv2DConv2D+model/vgg16/block4_conv2/Relu:activations:06model/vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2!
model/vgg16/block4_conv3/Conv2D?
/model/vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block4_conv3_biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block4_conv3/BiasAdd/ReadVariableOp?
 model/vgg16/block4_conv3/BiasAddBiasAdd(model/vgg16/block4_conv3/Conv2D:output:07model/vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/vgg16/block4_conv3/BiasAdd?
model/vgg16/block4_conv3/ReluRelu)model/vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/vgg16/block4_conv3/Relu?
model/vgg16/block4_pool/MaxPoolMaxPool+model/vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2!
model/vgg16/block4_pool/MaxPool?
.model/vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block5_conv1_conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype020
.model/vgg16/block5_conv1/Conv2D/ReadVariableOp?
model/vgg16/block5_conv1/Conv2DConv2D(model/vgg16/block4_pool/MaxPool:output:06model/vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2!
model/vgg16/block5_conv1/Conv2D?
/model/vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block5_conv1_biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block5_conv1/BiasAdd/ReadVariableOp?
 model/vgg16/block5_conv1/BiasAddBiasAdd(model/vgg16/block5_conv1/Conv2D:output:07model/vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/vgg16/block5_conv1/BiasAdd?
model/vgg16/block5_conv1/ReluRelu)model/vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/vgg16/block5_conv1/Relu?
.model/vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block5_conv2_conv2d_readvariableop_block5_conv2_kernel*(
_output_shapes
:??*
dtype020
.model/vgg16/block5_conv2/Conv2D/ReadVariableOp?
model/vgg16/block5_conv2/Conv2DConv2D+model/vgg16/block5_conv1/Relu:activations:06model/vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2!
model/vgg16/block5_conv2/Conv2D?
/model/vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block5_conv2_biasadd_readvariableop_block5_conv2_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block5_conv2/BiasAdd/ReadVariableOp?
 model/vgg16/block5_conv2/BiasAddBiasAdd(model/vgg16/block5_conv2/Conv2D:output:07model/vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/vgg16/block5_conv2/BiasAdd?
model/vgg16/block5_conv2/ReluRelu)model/vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/vgg16/block5_conv2/Relu?
.model/vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOpBmodel_vgg16_block5_conv3_conv2d_readvariableop_block5_conv3_kernel*(
_output_shapes
:??*
dtype020
.model/vgg16/block5_conv3/Conv2D/ReadVariableOp?
model/vgg16/block5_conv3/Conv2DConv2D+model/vgg16/block5_conv2/Relu:activations:06model/vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2!
model/vgg16/block5_conv3/Conv2D?
/model/vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOpAmodel_vgg16_block5_conv3_biasadd_readvariableop_block5_conv3_bias*
_output_shapes	
:?*
dtype021
/model/vgg16/block5_conv3/BiasAdd/ReadVariableOp?
 model/vgg16/block5_conv3/BiasAddBiasAdd(model/vgg16/block5_conv3/Conv2D:output:07model/vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/vgg16/block5_conv3/BiasAdd?
model/vgg16/block5_conv3/ReluRelu)model/vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/vgg16/block5_conv3/Relu?
model/vgg16/block5_pool/MaxPoolMaxPool+model/vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2!
model/vgg16/block5_pool/MaxPool?
;model/vgg16/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;model/vgg16/global_average_pooling2d/Mean/reduction_indices?
)model/vgg16/global_average_pooling2d/MeanMean(model/vgg16/block5_pool/MaxPool:output:0Dmodel/vgg16/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2+
)model/vgg16/global_average_pooling2d/Mean?
,model/sequential/dense/MatMul/ReadVariableOpReadVariableOp9model_sequential_dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
??*
dtype02.
,model/sequential/dense/MatMul/ReadVariableOp?
model/sequential/dense/MatMulMatMul2model/vgg16/global_average_pooling2d/Mean:output:04model/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/sequential/dense/MatMul?
-model/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp8model_sequential_dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:?*
dtype02/
-model/sequential/dense/BiasAdd/ReadVariableOp?
model/sequential/dense/BiasAddBiasAdd'model/sequential/dense/MatMul:product:05model/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
model/sequential/dense/BiasAdd?
model/sequential/dense/ReluRelu'model/sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/sequential/dense/Relu?
.model/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp=model_sequential_dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
??*
dtype020
.model/sequential/dense_1/MatMul/ReadVariableOp?
model/sequential/dense_1/MatMulMatMul)model/sequential/dense/Relu:activations:06model/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
model/sequential/dense_1/MatMul?
/model/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp<model_sequential_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:?*
dtype021
/model/sequential/dense_1/BiasAdd/ReadVariableOp?
 model/sequential/dense_1/BiasAddBiasAdd)model/sequential/dense_1/MatMul:product:07model/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 model/sequential/dense_1/BiasAdd?
model/sequential/dense_1/ReluRelu)model/sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/sequential/dense_1/Relu?
!model/sequential/dropout/IdentityIdentity+model/sequential/dense_1/Relu:activations:0*
T0*(
_output_shapes
:??????????2#
!model/sequential/dropout/Identity?
.model/sequential/dense_2/MatMul/ReadVariableOpReadVariableOp=model_sequential_dense_2_matmul_readvariableop_dense_2_kernel* 
_output_shapes
:
??*
dtype020
.model/sequential/dense_2/MatMul/ReadVariableOp?
model/sequential/dense_2/MatMulMatMul*model/sequential/dropout/Identity:output:06model/sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
model/sequential/dense_2/MatMul?
/model/sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp<model_sequential_dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes	
:?*
dtype021
/model/sequential/dense_2/BiasAdd/ReadVariableOp?
 model/sequential/dense_2/BiasAddBiasAdd)model/sequential/dense_2/MatMul:product:07model/sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 model/sequential/dense_2/BiasAdd?
.model/sequential/dense_3/MatMul/ReadVariableOpReadVariableOp=model_sequential_dense_3_matmul_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype020
.model/sequential/dense_3/MatMul/ReadVariableOp?
model/sequential/dense_3/MatMulMatMul)model/sequential/dense_2/BiasAdd:output:06model/sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
model/sequential/dense_3/MatMul?
/model/sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp<model_sequential_dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes
:*
dtype021
/model/sequential/dense_3/BiasAdd/ReadVariableOp?
 model/sequential/dense_3/BiasAddBiasAdd)model/sequential/dense_3/MatMul:product:07model/sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 model/sequential/dense_3/BiasAdd?
 model/sequential/dense_3/SoftmaxSoftmax)model/sequential/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2"
 model/sequential/dense_3/Softmax?
IdentityIdentity*model/sequential/dense_3/Softmax:softmax:0.^model/sequential/dense/BiasAdd/ReadVariableOp-^model/sequential/dense/MatMul/ReadVariableOp0^model/sequential/dense_1/BiasAdd/ReadVariableOp/^model/sequential/dense_1/MatMul/ReadVariableOp0^model/sequential/dense_2/BiasAdd/ReadVariableOp/^model/sequential/dense_2/MatMul/ReadVariableOp0^model/sequential/dense_3/BiasAdd/ReadVariableOp/^model/sequential/dense_3/MatMul/ReadVariableOp0^model/vgg16/block1_conv1/BiasAdd/ReadVariableOp/^model/vgg16/block1_conv1/Conv2D/ReadVariableOp0^model/vgg16/block1_conv2/BiasAdd/ReadVariableOp/^model/vgg16/block1_conv2/Conv2D/ReadVariableOp0^model/vgg16/block2_conv1/BiasAdd/ReadVariableOp/^model/vgg16/block2_conv1/Conv2D/ReadVariableOp0^model/vgg16/block2_conv2/BiasAdd/ReadVariableOp/^model/vgg16/block2_conv2/Conv2D/ReadVariableOp0^model/vgg16/block3_conv1/BiasAdd/ReadVariableOp/^model/vgg16/block3_conv1/Conv2D/ReadVariableOp0^model/vgg16/block3_conv2/BiasAdd/ReadVariableOp/^model/vgg16/block3_conv2/Conv2D/ReadVariableOp0^model/vgg16/block3_conv3/BiasAdd/ReadVariableOp/^model/vgg16/block3_conv3/Conv2D/ReadVariableOp0^model/vgg16/block4_conv1/BiasAdd/ReadVariableOp/^model/vgg16/block4_conv1/Conv2D/ReadVariableOp0^model/vgg16/block4_conv2/BiasAdd/ReadVariableOp/^model/vgg16/block4_conv2/Conv2D/ReadVariableOp0^model/vgg16/block4_conv3/BiasAdd/ReadVariableOp/^model/vgg16/block4_conv3/Conv2D/ReadVariableOp0^model/vgg16/block5_conv1/BiasAdd/ReadVariableOp/^model/vgg16/block5_conv1/Conv2D/ReadVariableOp0^model/vgg16/block5_conv2/BiasAdd/ReadVariableOp/^model/vgg16/block5_conv2/Conv2D/ReadVariableOp0^model/vgg16/block5_conv3/BiasAdd/ReadVariableOp/^model/vgg16/block5_conv3/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::2^
-model/sequential/dense/BiasAdd/ReadVariableOp-model/sequential/dense/BiasAdd/ReadVariableOp2\
,model/sequential/dense/MatMul/ReadVariableOp,model/sequential/dense/MatMul/ReadVariableOp2b
/model/sequential/dense_1/BiasAdd/ReadVariableOp/model/sequential/dense_1/BiasAdd/ReadVariableOp2`
.model/sequential/dense_1/MatMul/ReadVariableOp.model/sequential/dense_1/MatMul/ReadVariableOp2b
/model/sequential/dense_2/BiasAdd/ReadVariableOp/model/sequential/dense_2/BiasAdd/ReadVariableOp2`
.model/sequential/dense_2/MatMul/ReadVariableOp.model/sequential/dense_2/MatMul/ReadVariableOp2b
/model/sequential/dense_3/BiasAdd/ReadVariableOp/model/sequential/dense_3/BiasAdd/ReadVariableOp2`
.model/sequential/dense_3/MatMul/ReadVariableOp.model/sequential/dense_3/MatMul/ReadVariableOp2b
/model/vgg16/block1_conv1/BiasAdd/ReadVariableOp/model/vgg16/block1_conv1/BiasAdd/ReadVariableOp2`
.model/vgg16/block1_conv1/Conv2D/ReadVariableOp.model/vgg16/block1_conv1/Conv2D/ReadVariableOp2b
/model/vgg16/block1_conv2/BiasAdd/ReadVariableOp/model/vgg16/block1_conv2/BiasAdd/ReadVariableOp2`
.model/vgg16/block1_conv2/Conv2D/ReadVariableOp.model/vgg16/block1_conv2/Conv2D/ReadVariableOp2b
/model/vgg16/block2_conv1/BiasAdd/ReadVariableOp/model/vgg16/block2_conv1/BiasAdd/ReadVariableOp2`
.model/vgg16/block2_conv1/Conv2D/ReadVariableOp.model/vgg16/block2_conv1/Conv2D/ReadVariableOp2b
/model/vgg16/block2_conv2/BiasAdd/ReadVariableOp/model/vgg16/block2_conv2/BiasAdd/ReadVariableOp2`
.model/vgg16/block2_conv2/Conv2D/ReadVariableOp.model/vgg16/block2_conv2/Conv2D/ReadVariableOp2b
/model/vgg16/block3_conv1/BiasAdd/ReadVariableOp/model/vgg16/block3_conv1/BiasAdd/ReadVariableOp2`
.model/vgg16/block3_conv1/Conv2D/ReadVariableOp.model/vgg16/block3_conv1/Conv2D/ReadVariableOp2b
/model/vgg16/block3_conv2/BiasAdd/ReadVariableOp/model/vgg16/block3_conv2/BiasAdd/ReadVariableOp2`
.model/vgg16/block3_conv2/Conv2D/ReadVariableOp.model/vgg16/block3_conv2/Conv2D/ReadVariableOp2b
/model/vgg16/block3_conv3/BiasAdd/ReadVariableOp/model/vgg16/block3_conv3/BiasAdd/ReadVariableOp2`
.model/vgg16/block3_conv3/Conv2D/ReadVariableOp.model/vgg16/block3_conv3/Conv2D/ReadVariableOp2b
/model/vgg16/block4_conv1/BiasAdd/ReadVariableOp/model/vgg16/block4_conv1/BiasAdd/ReadVariableOp2`
.model/vgg16/block4_conv1/Conv2D/ReadVariableOp.model/vgg16/block4_conv1/Conv2D/ReadVariableOp2b
/model/vgg16/block4_conv2/BiasAdd/ReadVariableOp/model/vgg16/block4_conv2/BiasAdd/ReadVariableOp2`
.model/vgg16/block4_conv2/Conv2D/ReadVariableOp.model/vgg16/block4_conv2/Conv2D/ReadVariableOp2b
/model/vgg16/block4_conv3/BiasAdd/ReadVariableOp/model/vgg16/block4_conv3/BiasAdd/ReadVariableOp2`
.model/vgg16/block4_conv3/Conv2D/ReadVariableOp.model/vgg16/block4_conv3/Conv2D/ReadVariableOp2b
/model/vgg16/block5_conv1/BiasAdd/ReadVariableOp/model/vgg16/block5_conv1/BiasAdd/ReadVariableOp2`
.model/vgg16/block5_conv1/Conv2D/ReadVariableOp.model/vgg16/block5_conv1/Conv2D/ReadVariableOp2b
/model/vgg16/block5_conv2/BiasAdd/ReadVariableOp/model/vgg16/block5_conv2/BiasAdd/ReadVariableOp2`
.model/vgg16/block5_conv2/Conv2D/ReadVariableOp.model/vgg16/block5_conv2/Conv2D/ReadVariableOp2b
/model/vgg16/block5_conv3/BiasAdd/ReadVariableOp/model/vgg16/block5_conv3/BiasAdd/ReadVariableOp2`
.model/vgg16/block5_conv3/Conv2D/ReadVariableOp.model/vgg16/block5_conv3/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1965
dense_input
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
dense_3_dense_3_kernel
dense_3_dense_3_bias
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_dense_kerneldense_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_18362
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_18592!
dense_1/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_18892
dropout/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_19122!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_dense_3_kerneldense_3_dense_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_19352!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?

$__inference_dense_layer_call_fn_3355

inputs
dense_kernel

dense_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_18362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_1198

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv1_layer_call_and_return_conditional_losses_3240

inputs-
)conv2d_readvariableop_block4_conv1_kernel,
(biasadd_readvariableop_block4_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_dense_3_layer_call_and_return_conditional_losses_3428

inputs(
$matmul_readvariableop_dense_3_kernel'
#biasadd_readvariableop_dense_3_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_3_kernel*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_3_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
&__inference_dropout_layer_call_fn_3395

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_18842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv2_layer_call_and_return_conditional_losses_1563

inputs-
)conv2d_readvariableop_block5_conv2_kernel,
(biasadd_readvariableop_block5_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block5_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block5_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block1_conv1_layer_call_and_return_conditional_losses_3114

inputs-
)conv2d_readvariableop_block1_conv1_kernel,
(biasadd_readvariableop_block1_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_1190

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_2395
input_2
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias
block5_conv2_kernel
block5_conv2_bias
block5_conv3_kernel
block5_conv3_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biasblock5_conv2_kernelblock5_conv2_biasblock5_conv3_kernelblock5_conv3_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_11672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_1249

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv1_layer_call_and_return_conditional_losses_1465

inputs-
)conv2d_readvariableop_block4_conv1_kernel,
(biasadd_readvariableop_block4_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_2026
dense_input
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_20152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:??????????
%
_user_specified_namedense_input
?
?
F__inference_block4_conv2_layer_call_and_return_conditional_losses_3258

inputs-
)conv2d_readvariableop_block4_conv2_kernel,
(biasadd_readvariableop_block4_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_block5_conv1_layer_call_fn_3301

inputs
block5_conv1_kernel
block5_conv1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock5_conv1_kernelblock5_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_15402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block1_conv1_layer_call_and_return_conditional_losses_1286

inputs-
)conv2d_readvariableop_block1_conv1_kernel,
(biasadd_readvariableop_block1_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_vgg16_layer_call_fn_1821
input_1
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias
block5_conv2_kernel
block5_conv2_bias
block5_conv3_kernel
block5_conv3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biasblock5_conv2_kernelblock5_conv2_biasblock5_conv3_kernelblock5_conv3_bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_17922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_1241

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_1207

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv1_layer_call_and_return_conditional_losses_3294

inputs-
)conv2d_readvariableop_block5_conv1_kernel,
(biasadd_readvariableop_block5_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_2_layer_call_and_return_conditional_losses_3410

inputs(
$matmul_readvariableop_dense_2_kernel'
#biasadd_readvariableop_dense_2_bias
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_3090

inputs
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
dense_3_kernel
dense_3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_biasdense_3_kerneldense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_19852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_block3_pool_layer_call_fn_1218

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_12152
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?	
?__inference_model_layer_call_and_return_conditional_losses_2155
input_2
vgg16_block1_conv1_kernel
vgg16_block1_conv1_bias
vgg16_block1_conv2_kernel
vgg16_block1_conv2_bias
vgg16_block2_conv1_kernel
vgg16_block2_conv1_bias
vgg16_block2_conv2_kernel
vgg16_block2_conv2_bias
vgg16_block3_conv1_kernel
vgg16_block3_conv1_bias
vgg16_block3_conv2_kernel
vgg16_block3_conv2_bias
vgg16_block3_conv3_kernel
vgg16_block3_conv3_bias
vgg16_block4_conv1_kernel
vgg16_block4_conv1_bias
vgg16_block4_conv2_kernel
vgg16_block4_conv2_bias
vgg16_block4_conv3_kernel
vgg16_block4_conv3_bias
vgg16_block5_conv1_kernel
vgg16_block5_conv1_bias
vgg16_block5_conv2_kernel
vgg16_block5_conv2_bias
vgg16_block5_conv3_kernel
vgg16_block5_conv3_bias
sequential_dense_kernel
sequential_dense_bias
sequential_dense_1_kernel
sequential_dense_1_bias
sequential_dense_2_kernel
sequential_dense_2_bias
sequential_dense_3_kernel
sequential_dense_3_bias
identity??"sequential/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?
vgg16/StatefulPartitionedCallStatefulPartitionedCallinput_2vgg16_block1_conv1_kernelvgg16_block1_conv1_biasvgg16_block1_conv2_kernelvgg16_block1_conv2_biasvgg16_block2_conv1_kernelvgg16_block2_conv1_biasvgg16_block2_conv2_kernelvgg16_block2_conv2_biasvgg16_block3_conv1_kernelvgg16_block3_conv1_biasvgg16_block3_conv2_kernelvgg16_block3_conv2_biasvgg16_block3_conv3_kernelvgg16_block3_conv3_biasvgg16_block4_conv1_kernelvgg16_block4_conv1_biasvgg16_block4_conv2_kernelvgg16_block4_conv2_biasvgg16_block4_conv3_kernelvgg16_block4_conv3_biasvgg16_block5_conv1_kernelvgg16_block5_conv1_biasvgg16_block5_conv2_kernelvgg16_block5_conv2_biasvgg16_block5_conv3_kernelvgg16_block5_conv3_bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_17122
vgg16/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0sequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_biassequential_dense_3_kernelsequential_dense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_19852$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_2
?`
?
?__inference_vgg16_layer_call_and_return_conditional_losses_1712

inputs$
 block1_conv1_block1_conv1_kernel"
block1_conv1_block1_conv1_bias$
 block1_conv2_block1_conv2_kernel"
block1_conv2_block1_conv2_bias$
 block2_conv1_block2_conv1_kernel"
block2_conv1_block2_conv1_bias$
 block2_conv2_block2_conv2_kernel"
block2_conv2_block2_conv2_bias$
 block3_conv1_block3_conv1_kernel"
block3_conv1_block3_conv1_bias$
 block3_conv2_block3_conv2_kernel"
block3_conv2_block3_conv2_bias$
 block3_conv3_block3_conv3_kernel"
block3_conv3_block3_conv3_bias$
 block4_conv1_block4_conv1_kernel"
block4_conv1_block4_conv1_bias$
 block4_conv2_block4_conv2_kernel"
block4_conv2_block4_conv2_bias$
 block4_conv3_block4_conv3_kernel"
block4_conv3_block4_conv3_bias$
 block5_conv1_block5_conv1_kernel"
block5_conv1_block5_conv1_bias$
 block5_conv2_block5_conv2_kernel"
block5_conv2_block5_conv2_bias$
 block5_conv3_block5_conv3_kernel"
block5_conv3_block5_conv3_bias
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputs block1_conv1_block1_conv1_kernelblock1_conv1_block1_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_12862&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0 block1_conv2_block1_conv2_kernelblock1_conv2_block1_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_13092&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_11812
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0 block2_conv1_block2_conv1_kernelblock2_conv1_block2_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_13382&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0 block2_conv2_block2_conv2_kernelblock2_conv2_block2_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_13612&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_11982
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0 block3_conv1_block3_conv1_kernelblock3_conv1_block3_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_13902&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0 block3_conv2_block3_conv2_kernelblock3_conv2_block3_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_14132&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0 block3_conv3_block3_conv3_kernelblock3_conv3_block3_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_14362&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_12152
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0 block4_conv1_block4_conv1_kernelblock4_conv1_block4_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_14652&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0 block4_conv2_block4_conv2_kernelblock4_conv2_block4_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_14882&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0 block4_conv3_block4_conv3_kernelblock4_conv3_block4_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_15112&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_12322
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0 block5_conv1_block5_conv1_kernelblock5_conv1_block5_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_15402&
$block5_conv1/StatefulPartitionedCall?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0 block5_conv2_block5_conv2_kernelblock5_conv2_block5_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_15632&
$block5_conv2/StatefulPartitionedCall?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0 block5_conv3_block5_conv3_kernelblock5_conv3_block5_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_15862&
$block5_conv3/StatefulPartitionedCall?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_12492
block5_pool/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall$block5_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_12682*
(global_average_pooling2d/PartitionedCall?
IdentityIdentity1global_average_pooling2d/PartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?	
?__inference_model_layer_call_and_return_conditional_losses_2317

inputs
vgg16_block1_conv1_kernel
vgg16_block1_conv1_bias
vgg16_block1_conv2_kernel
vgg16_block1_conv2_bias
vgg16_block2_conv1_kernel
vgg16_block2_conv1_bias
vgg16_block2_conv2_kernel
vgg16_block2_conv2_bias
vgg16_block3_conv1_kernel
vgg16_block3_conv1_bias
vgg16_block3_conv2_kernel
vgg16_block3_conv2_bias
vgg16_block3_conv3_kernel
vgg16_block3_conv3_bias
vgg16_block4_conv1_kernel
vgg16_block4_conv1_bias
vgg16_block4_conv2_kernel
vgg16_block4_conv2_bias
vgg16_block4_conv3_kernel
vgg16_block4_conv3_bias
vgg16_block5_conv1_kernel
vgg16_block5_conv1_bias
vgg16_block5_conv2_kernel
vgg16_block5_conv2_bias
vgg16_block5_conv3_kernel
vgg16_block5_conv3_bias
sequential_dense_kernel
sequential_dense_bias
sequential_dense_1_kernel
sequential_dense_1_bias
sequential_dense_2_kernel
sequential_dense_2_bias
sequential_dense_3_kernel
sequential_dense_3_bias
identity??"sequential/StatefulPartitionedCall?vgg16/StatefulPartitionedCall?
vgg16/StatefulPartitionedCallStatefulPartitionedCallinputsvgg16_block1_conv1_kernelvgg16_block1_conv1_biasvgg16_block1_conv2_kernelvgg16_block1_conv2_biasvgg16_block2_conv1_kernelvgg16_block2_conv1_biasvgg16_block2_conv2_kernelvgg16_block2_conv2_biasvgg16_block3_conv1_kernelvgg16_block3_conv1_biasvgg16_block3_conv2_kernelvgg16_block3_conv2_biasvgg16_block3_conv3_kernelvgg16_block3_conv3_biasvgg16_block4_conv1_kernelvgg16_block4_conv1_biasvgg16_block4_conv2_kernelvgg16_block4_conv2_biasvgg16_block4_conv3_kernelvgg16_block4_conv3_biasvgg16_block5_conv1_kernelvgg16_block5_conv1_biasvgg16_block5_conv2_kernelvgg16_block5_conv2_biasvgg16_block5_conv3_kernelvgg16_block5_conv3_bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_17922
vgg16/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&vgg16/StatefulPartitionedCall:output:0sequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_biassequential_dense_3_kernelsequential_dense_3_bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_20152$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
+__inference_block3_conv3_layer_call_fn_3229

inputs
block3_conv3_kernel
block3_conv3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock3_conv3_kernelblock3_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_14362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????88?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????88?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?p
?
__inference__traced_save_3635
file_prefix3
/savev2_training_adamax_iter_read_readvariableop	5
1savev2_training_adamax_beta_1_read_readvariableop5
1savev2_training_adamax_beta_2_read_readvariableop4
0savev2_training_adamax_decay_read_readvariableop<
8savev2_training_adamax_learning_rate_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_training_adamax_dense_kernel_m_read_readvariableop;
7savev2_training_adamax_dense_bias_m_read_readvariableop?
;savev2_training_adamax_dense_1_kernel_m_read_readvariableop=
9savev2_training_adamax_dense_1_bias_m_read_readvariableop?
;savev2_training_adamax_dense_2_kernel_m_read_readvariableop=
9savev2_training_adamax_dense_2_bias_m_read_readvariableop?
;savev2_training_adamax_dense_3_kernel_m_read_readvariableop=
9savev2_training_adamax_dense_3_bias_m_read_readvariableop=
9savev2_training_adamax_dense_kernel_v_read_readvariableop;
7savev2_training_adamax_dense_bias_v_read_readvariableop?
;savev2_training_adamax_dense_1_kernel_v_read_readvariableop=
9savev2_training_adamax_dense_1_bias_v_read_readvariableop?
;savev2_training_adamax_dense_2_kernel_v_read_readvariableop=
9savev2_training_adamax_dense_2_bias_v_read_readvariableop?
;savev2_training_adamax_dense_3_kernel_v_read_readvariableop=
9savev2_training_adamax_dense_3_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_training_adamax_iter_read_readvariableop1savev2_training_adamax_beta_1_read_readvariableop1savev2_training_adamax_beta_2_read_readvariableop0savev2_training_adamax_decay_read_readvariableop8savev2_training_adamax_learning_rate_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_training_adamax_dense_kernel_m_read_readvariableop7savev2_training_adamax_dense_bias_m_read_readvariableop;savev2_training_adamax_dense_1_kernel_m_read_readvariableop9savev2_training_adamax_dense_1_bias_m_read_readvariableop;savev2_training_adamax_dense_2_kernel_m_read_readvariableop9savev2_training_adamax_dense_2_bias_m_read_readvariableop;savev2_training_adamax_dense_3_kernel_m_read_readvariableop9savev2_training_adamax_dense_3_bias_m_read_readvariableop9savev2_training_adamax_dense_kernel_v_read_readvariableop7savev2_training_adamax_dense_bias_v_read_readvariableop;savev2_training_adamax_dense_1_kernel_v_read_readvariableop9savev2_training_adamax_dense_1_bias_v_read_readvariableop;savev2_training_adamax_dense_2_kernel_v_read_readvariableop9savev2_training_adamax_dense_2_bias_v_read_readvariableop;savev2_training_adamax_dense_3_kernel_v_read_readvariableop9savev2_training_adamax_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *J
dtypes@
>2<	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :@:@:@@:@:@?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:
??:?:
??:?:
??:?:	?:: : : : :
??:?:
??:?:
??:?:	?::
??:?:
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 	

_output_shapes
:@:-
)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:& "
 
_output_shapes
:
??:!!

_output_shapes	
:?:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:&$"
 
_output_shapes
:
??:!%

_output_shapes	
:?:%&!

_output_shapes
:	?: '

_output_shapes
::(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :&,"
 
_output_shapes
:
??:!-

_output_shapes	
:?:&."
 
_output_shapes
:
??:!/

_output_shapes	
:?:&0"
 
_output_shapes
:
??:!1

_output_shapes	
:?:%2!

_output_shapes
:	?: 3

_output_shapes
::&4"
 
_output_shapes
:
??:!5

_output_shapes	
:?:&6"
 
_output_shapes
:
??:!7

_output_shapes	
:?:&8"
 
_output_shapes
:
??:!9

_output_shapes	
:?:%:!

_output_shapes
:	?: ;

_output_shapes
::<

_output_shapes
: 
?
?
+__inference_block4_conv1_layer_call_fn_3247

inputs
block4_conv1_kernel
block4_conv1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock4_conv1_kernelblock4_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_14652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
?__inference_vgg16_layer_call_and_return_conditional_losses_2842

inputs:
6block1_conv1_conv2d_readvariableop_block1_conv1_kernel9
5block1_conv1_biasadd_readvariableop_block1_conv1_bias:
6block1_conv2_conv2d_readvariableop_block1_conv2_kernel9
5block1_conv2_biasadd_readvariableop_block1_conv2_bias:
6block2_conv1_conv2d_readvariableop_block2_conv1_kernel9
5block2_conv1_biasadd_readvariableop_block2_conv1_bias:
6block2_conv2_conv2d_readvariableop_block2_conv2_kernel9
5block2_conv2_biasadd_readvariableop_block2_conv2_bias:
6block3_conv1_conv2d_readvariableop_block3_conv1_kernel9
5block3_conv1_biasadd_readvariableop_block3_conv1_bias:
6block3_conv2_conv2d_readvariableop_block3_conv2_kernel9
5block3_conv2_biasadd_readvariableop_block3_conv2_bias:
6block3_conv3_conv2d_readvariableop_block3_conv3_kernel9
5block3_conv3_biasadd_readvariableop_block3_conv3_bias:
6block4_conv1_conv2d_readvariableop_block4_conv1_kernel9
5block4_conv1_biasadd_readvariableop_block4_conv1_bias:
6block4_conv2_conv2d_readvariableop_block4_conv2_kernel9
5block4_conv2_biasadd_readvariableop_block4_conv2_bias:
6block4_conv3_conv2d_readvariableop_block4_conv3_kernel9
5block4_conv3_biasadd_readvariableop_block4_conv3_bias:
6block5_conv1_conv2d_readvariableop_block5_conv1_kernel9
5block5_conv1_biasadd_readvariableop_block5_conv1_bias:
6block5_conv2_conv2d_readvariableop_block5_conv2_kernel9
5block5_conv2_biasadd_readvariableop_block5_conv2_bias:
6block5_conv3_conv2d_readvariableop_block5_conv3_kernel9
5block5_conv3_biasadd_readvariableop_block5_conv3_bias
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp6block1_conv1_conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp5block1_conv1_biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv1/Relu?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp6block1_conv2_conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp5block1_conv2_biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp6block2_conv1_conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp5block2_conv1_biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv1/Relu?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp6block2_conv2_conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp5block2_conv2_biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp6block3_conv1_conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp5block3_conv1_biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv1/Relu?
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp6block3_conv2_conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv2/Conv2D?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp5block3_conv2_biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv2/BiasAdd?
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv2/Relu?
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp6block3_conv3_conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
2
block3_conv3/Conv2D?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp5block3_conv3_biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?2
block3_conv3/BiasAdd?
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88?2
block3_conv3/Relu?
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp6block4_conv1_conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv1/Conv2D?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp5block4_conv1_biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv1/BiasAdd?
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv1/Relu?
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp6block4_conv2_conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv2/Conv2D?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp5block4_conv2_biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv2/BiasAdd?
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv2/Relu?
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp6block4_conv3_conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block4_conv3/Conv2D?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp5block4_conv3_biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block4_conv3/BiasAdd?
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv3/Relu?
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp6block5_conv1_conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv1/Conv2D?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp5block5_conv1_biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv1/BiasAdd?
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv1/Relu?
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp6block5_conv2_conv2d_readvariableop_block5_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block5_conv2/Conv2D/ReadVariableOp?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv2/Conv2D?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp5block5_conv2_biasadd_readvariableop_block5_conv2_bias*
_output_shapes	
:?*
dtype02%
#block5_conv2/BiasAdd/ReadVariableOp?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv2/BiasAdd?
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv2/Relu?
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp6block5_conv3_conv2d_readvariableop_block5_conv3_kernel*(
_output_shapes
:??*
dtype02$
"block5_conv3/Conv2D/ReadVariableOp?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
block5_conv3/Conv2D?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp5block5_conv3_biasadd_readvariableop_block5_conv3_bias*
_output_shapes	
:?*
dtype02%
#block5_conv3/BiasAdd/ReadVariableOp?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
block5_conv3/BiasAdd?
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv3/Relu?
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
block5_pool/MaxPool?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeanblock5_pool/MaxPool:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
global_average_pooling2d/Mean?
IdentityIdentity&global_average_pooling2d/Mean:output:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_2:
serving_default_input_2:0???????????>

sequential0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"??
_tf_keras_network??{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["block5_pool", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["global_average_pooling2d", 0, 0]]}, "name": "vgg16", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["vgg16", 1, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["sequential", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["block5_pool", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["global_average_pooling2d", 0, 0]]}, "name": "vgg16", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["vgg16", 1, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["sequential", 1, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adamax", "config": {"name": "Adamax", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
??

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
layer-18
layer-19
	variables
regularization_losses
 trainable_variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "vgg16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["block5_pool", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["global_average_pooling2d", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "vgg16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv2", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv3", "inbound_nodes": [[["block5_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block5_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block5_pool", "inbound_nodes": [[["block5_conv3", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["block5_pool", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["global_average_pooling2d", 0, 0]]}}}
?-
"layer_with_weights-0
"layer-0
#layer_with_weights-1
#layer-1
$layer-2
%layer_with_weights-2
%layer-3
&layer_with_weights-3
&layer-4
	optimizer
'	variables
(regularization_losses
)trainable_variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?+
_tf_keras_sequential?*{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adamax", "config": {"name": "Adamax", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?
+iter

,beta_1

-beta_2
	.decay
/learning_rateJm?Km?Lm?Mm?Nm?Om?Pm?Qm?Jv?Kv?Lv?Mv?Nv?Ov?Pv?Qv?"
	optimizer
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23
H24
I25
J26
K27
L28
M29
N30
O31
P32
Q33"
trackable_list_wrapper
 "
trackable_list_wrapper
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23
H24
I25
J26
K27
L28
M29
N30
O31
P32
Q33"
trackable_list_wrapper
?
Rnon_trainable_variables
	variables
Smetrics
regularization_losses

Tlayers
trainable_variables
Ulayer_regularization_losses
Vlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?	

0kernel
1bias
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 3]}}
?	

2kernel
3bias
[	variables
\regularization_losses
]trainable_variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block1_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 64]}}
?
_	variables
`regularization_losses
atrainable_variables
b	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block1_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

4kernel
5bias
c	variables
dregularization_losses
etrainable_variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2_conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 112, 112, 64]}}
?


6kernel
7bias
g	variables
hregularization_losses
itrainable_variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block2_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 112, 112, 128]}}
?
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block2_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

8kernel
9bias
o	variables
pregularization_losses
qtrainable_variables
r	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3_conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 128]}}
?	

:kernel
;bias
s	variables
tregularization_losses
utrainable_variables
v	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block3_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3_conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 256]}}
?	

<kernel
=bias
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block3_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3_conv3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56, 56, 256]}}
?
{	variables
|regularization_losses
}trainable_variables
~	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block3_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


>kernel
?bias
	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 256]}}
?


@kernel
Abias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block4_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 512]}}
?


Bkernel
Cbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block4_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 512]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block4_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


Dkernel
Ebias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block5_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block5_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 512]}}
?


Fkernel
Gbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block5_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block5_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 512]}}
?


Hkernel
Ibias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block5_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block5_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 512]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block5_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block5_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23
H24
I25"
trackable_list_wrapper
 "
trackable_list_wrapper
?
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
B18
C19
D20
E21
F22
G23
H24
I25"
trackable_list_wrapper
?
?non_trainable_variables
	variables
?metrics
regularization_losses
?layers
 trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Jkernel
Kbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?

Lkernel
Mbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

Nkernel
Obias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Pkernel
Qbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
X
J0
K1
L2
M3
N4
O5
P6
Q7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
J0
K1
L2
M3
N4
O5
P6
Q7"
trackable_list_wrapper
?
?non_trainable_variables
'	variables
?metrics
(regularization_losses
?layers
)trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adamax/iter
 : (2training/Adamax/beta_1
 : (2training/Adamax/beta_2
: (2training/Adamax/decay
':% (2training/Adamax/learning_rate
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@?2block2_conv1/kernel
 :?2block2_conv1/bias
/:-??2block2_conv2/kernel
 :?2block2_conv2/bias
/:-??2block3_conv1/kernel
 :?2block3_conv1/bias
/:-??2block3_conv2/kernel
 :?2block3_conv2/bias
/:-??2block3_conv3/kernel
 :?2block3_conv3/bias
/:-??2block4_conv1/kernel
 :?2block4_conv1/bias
/:-??2block4_conv2/kernel
 :?2block4_conv2/bias
/:-??2block4_conv3/kernel
 :?2block4_conv3/bias
/:-??2block5_conv1/kernel
 :?2block5_conv1/bias
/:-??2block5_conv2/kernel
 :?2block5_conv2/bias
/:-??2block5_conv3/kernel
 :?2block5_conv3/bias
 :
??2dense/kernel
:?2
dense/bias
": 
??2dense_1/kernel
:?2dense_1/bias
": 
??2dense_2/kernel
:?2dense_2/bias
!:	?2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
?non_trainable_variables
W	variables
?metrics
Xregularization_losses
?layers
Ytrainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
?non_trainable_variables
[	variables
?metrics
\regularization_losses
?layers
]trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
_	variables
?metrics
`regularization_losses
?layers
atrainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
?non_trainable_variables
c	variables
?metrics
dregularization_losses
?layers
etrainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
?non_trainable_variables
g	variables
?metrics
hregularization_losses
?layers
itrainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
k	variables
?metrics
lregularization_losses
?layers
mtrainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
?non_trainable_variables
o	variables
?metrics
pregularization_losses
?layers
qtrainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
?non_trainable_variables
s	variables
?metrics
tregularization_losses
?layers
utrainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
?non_trainable_variables
w	variables
?metrics
xregularization_losses
?layers
ytrainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
{	variables
?metrics
|regularization_losses
?layers
}trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
?
?non_trainable_variables
	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
?non_trainable_variables
?	variables
?metrics
?regularization_losses
?layers
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
C
"0
#1
$2
%3
&4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total_1
:  (2count_1
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
0:.
??2training/Adamax/dense/kernel/m
):'?2training/Adamax/dense/bias/m
2:0
??2 training/Adamax/dense_1/kernel/m
+:)?2training/Adamax/dense_1/bias/m
2:0
??2 training/Adamax/dense_2/kernel/m
+:)?2training/Adamax/dense_2/bias/m
1:/	?2 training/Adamax/dense_3/kernel/m
*:(2training/Adamax/dense_3/bias/m
0:.
??2training/Adamax/dense/kernel/v
):'?2training/Adamax/dense/bias/v
2:0
??2 training/Adamax/dense_1/kernel/v
+:)?2training/Adamax/dense_1/bias/v
2:0
??2 training/Adamax/dense_2/kernel/v
+:)?2training/Adamax/dense_2/bias/v
1:/	?2 training/Adamax/dense_3/kernel/v
*:(2training/Adamax/dense_3/bias/v
?2?
?__inference_model_layer_call_and_return_conditional_losses_2662
?__inference_model_layer_call_and_return_conditional_losses_2532
?__inference_model_layer_call_and_return_conditional_losses_2155
?__inference_model_layer_call_and_return_conditional_losses_2195?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_model_layer_call_fn_2740
$__inference_model_layer_call_fn_2275
$__inference_model_layer_call_fn_2701
$__inference_model_layer_call_fn_2354?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_1167?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_2???????????
?2?
?__inference_vgg16_layer_call_and_return_conditional_losses_2842
?__inference_vgg16_layer_call_and_return_conditional_losses_1611
?__inference_vgg16_layer_call_and_return_conditional_losses_1660
?__inference_vgg16_layer_call_and_return_conditional_losses_2944?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_vgg16_layer_call_fn_1741
$__inference_vgg16_layer_call_fn_3006
$__inference_vgg16_layer_call_fn_1821
$__inference_vgg16_layer_call_fn_2975?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_1948
D__inference_sequential_layer_call_and_return_conditional_losses_3045
D__inference_sequential_layer_call_and_return_conditional_losses_3077
D__inference_sequential_layer_call_and_return_conditional_losses_1965?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_sequential_layer_call_fn_3090
)__inference_sequential_layer_call_fn_3103
)__inference_sequential_layer_call_fn_2026
)__inference_sequential_layer_call_fn_1996?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference_signature_wrapper_2395input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block1_conv1_layer_call_and_return_conditional_losses_3114?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block1_conv1_layer_call_fn_3121?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_3132?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block1_conv2_layer_call_fn_3139?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_block1_pool_layer_call_and_return_conditional_losses_1173?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_block1_pool_layer_call_fn_1184?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_block2_conv1_layer_call_and_return_conditional_losses_3150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block2_conv1_layer_call_fn_3157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block2_conv2_layer_call_and_return_conditional_losses_3168?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block2_conv2_layer_call_fn_3175?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_block2_pool_layer_call_and_return_conditional_losses_1190?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_block2_pool_layer_call_fn_1201?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_block3_conv1_layer_call_and_return_conditional_losses_3186?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block3_conv1_layer_call_fn_3193?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_conv2_layer_call_and_return_conditional_losses_3204?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block3_conv2_layer_call_fn_3211?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_conv3_layer_call_and_return_conditional_losses_3222?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block3_conv3_layer_call_fn_3229?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_block3_pool_layer_call_and_return_conditional_losses_1207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_block3_pool_layer_call_fn_1218?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_block4_conv1_layer_call_and_return_conditional_losses_3240?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block4_conv1_layer_call_fn_3247?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_conv2_layer_call_and_return_conditional_losses_3258?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block4_conv2_layer_call_fn_3265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_conv3_layer_call_and_return_conditional_losses_3276?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block4_conv3_layer_call_fn_3283?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_block4_pool_layer_call_and_return_conditional_losses_1224?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_block4_pool_layer_call_fn_1235?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_block5_conv1_layer_call_and_return_conditional_losses_3294?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block5_conv1_layer_call_fn_3301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block5_conv2_layer_call_and_return_conditional_losses_3312?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block5_conv2_layer_call_fn_3319?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block5_conv3_layer_call_and_return_conditional_losses_3330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block5_conv3_layer_call_fn_3337?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_block5_pool_layer_call_and_return_conditional_losses_1241?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_block5_pool_layer_call_fn_1252?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1259?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
7__inference_global_average_pooling2d_layer_call_fn_1271?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
?__inference_dense_layer_call_and_return_conditional_losses_3348?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_3355?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_3366?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_3373?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_3385
A__inference_dropout_layer_call_and_return_conditional_losses_3390?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dropout_layer_call_fn_3395
&__inference_dropout_layer_call_fn_3400?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_3410?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_3417?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_3_layer_call_and_return_conditional_losses_3428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_3_layer_call_fn_3435?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_1167?"0123456789:;<=>?@ABCDEFGHIJKLMNOPQ:?7
0?-
+?(
input_2???????????
? "7?4
2

sequential$?!

sequential??????????
F__inference_block1_conv1_layer_call_and_return_conditional_losses_3114p019?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????@
? ?
+__inference_block1_conv1_layer_call_fn_3121c019?6
/?,
*?'
inputs???????????
? ""????????????@?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_3132p239?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
+__inference_block1_conv2_layer_call_fn_3139c239?6
/?,
*?'
inputs???????????@
? ""????????????@?
E__inference_block1_pool_layer_call_and_return_conditional_losses_1173?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_block1_pool_layer_call_fn_1184?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_block2_conv1_layer_call_and_return_conditional_losses_3150m457?4
-?*
(?%
inputs?????????pp@
? ".?+
$?!
0?????????pp?
? ?
+__inference_block2_conv1_layer_call_fn_3157`457?4
-?*
(?%
inputs?????????pp@
? "!??????????pp??
F__inference_block2_conv2_layer_call_and_return_conditional_losses_3168n678?5
.?+
)?&
inputs?????????pp?
? ".?+
$?!
0?????????pp?
? ?
+__inference_block2_conv2_layer_call_fn_3175a678?5
.?+
)?&
inputs?????????pp?
? "!??????????pp??
E__inference_block2_pool_layer_call_and_return_conditional_losses_1190?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_block2_pool_layer_call_fn_1201?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_block3_conv1_layer_call_and_return_conditional_losses_3186n898?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
+__inference_block3_conv1_layer_call_fn_3193a898?5
.?+
)?&
inputs?????????88?
? "!??????????88??
F__inference_block3_conv2_layer_call_and_return_conditional_losses_3204n:;8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
+__inference_block3_conv2_layer_call_fn_3211a:;8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
F__inference_block3_conv3_layer_call_and_return_conditional_losses_3222n<=8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
+__inference_block3_conv3_layer_call_fn_3229a<=8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
E__inference_block3_pool_layer_call_and_return_conditional_losses_1207?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_block3_pool_layer_call_fn_1218?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_block4_conv1_layer_call_and_return_conditional_losses_3240n>?8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block4_conv1_layer_call_fn_3247a>?8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block4_conv2_layer_call_and_return_conditional_losses_3258n@A8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block4_conv2_layer_call_fn_3265a@A8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block4_conv3_layer_call_and_return_conditional_losses_3276nBC8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block4_conv3_layer_call_fn_3283aBC8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_block4_pool_layer_call_and_return_conditional_losses_1224?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_block4_pool_layer_call_fn_1235?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_block5_conv1_layer_call_and_return_conditional_losses_3294nDE8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block5_conv1_layer_call_fn_3301aDE8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block5_conv2_layer_call_and_return_conditional_losses_3312nFG8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block5_conv2_layer_call_fn_3319aFG8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block5_conv3_layer_call_and_return_conditional_losses_3330nHI8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block5_conv3_layer_call_fn_3337aHI8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_block5_pool_layer_call_and_return_conditional_losses_1241?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_block5_pool_layer_call_fn_1252?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_dense_1_layer_call_and_return_conditional_losses_3366^LM0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_1_layer_call_fn_3373QLM0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_2_layer_call_and_return_conditional_losses_3410^NO0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_2_layer_call_fn_3417QNO0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_3_layer_call_and_return_conditional_losses_3428]PQ0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_dense_3_layer_call_fn_3435PPQ0?-
&?#
!?
inputs??????????
? "???????????
?__inference_dense_layer_call_and_return_conditional_losses_3348^JK0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? y
$__inference_dense_layer_call_fn_3355QJK0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dropout_layer_call_and_return_conditional_losses_3385^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_3390^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? {
&__inference_dropout_layer_call_fn_3395Q4?1
*?'
!?
inputs??????????
p
? "???????????{
&__inference_dropout_layer_call_fn_3400Q4?1
*?'
!?
inputs??????????
p 
? "????????????
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1259?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
7__inference_global_average_pooling2d_layer_call_fn_1271wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
?__inference_model_layer_call_and_return_conditional_losses_2155?"0123456789:;<=>?@ABCDEFGHIJKLMNOPQB??
8?5
+?(
input_2???????????
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_2195?"0123456789:;<=>?@ABCDEFGHIJKLMNOPQB??
8?5
+?(
input_2???????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_2532?"0123456789:;<=>?@ABCDEFGHIJKLMNOPQA?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_2662?"0123456789:;<=>?@ABCDEFGHIJKLMNOPQA?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
$__inference_model_layer_call_fn_2275?"0123456789:;<=>?@ABCDEFGHIJKLMNOPQB??
8?5
+?(
input_2???????????
p

 
? "???????????
$__inference_model_layer_call_fn_2354?"0123456789:;<=>?@ABCDEFGHIJKLMNOPQB??
8?5
+?(
input_2???????????
p 

 
? "???????????
$__inference_model_layer_call_fn_2701?"0123456789:;<=>?@ABCDEFGHIJKLMNOPQA?>
7?4
*?'
inputs???????????
p

 
? "???????????
$__inference_model_layer_call_fn_2740?"0123456789:;<=>?@ABCDEFGHIJKLMNOPQA?>
7?4
*?'
inputs???????????
p 

 
? "???????????
D__inference_sequential_layer_call_and_return_conditional_losses_1948pJKLMNOPQ=?:
3?0
&?#
dense_input??????????
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1965pJKLMNOPQ=?:
3?0
&?#
dense_input??????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_3045kJKLMNOPQ8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_3077kJKLMNOPQ8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
)__inference_sequential_layer_call_fn_1996cJKLMNOPQ=?:
3?0
&?#
dense_input??????????
p

 
? "???????????
)__inference_sequential_layer_call_fn_2026cJKLMNOPQ=?:
3?0
&?#
dense_input??????????
p 

 
? "???????????
)__inference_sequential_layer_call_fn_3090^JKLMNOPQ8?5
.?+
!?
inputs??????????
p

 
? "???????????
)__inference_sequential_layer_call_fn_3103^JKLMNOPQ8?5
.?+
!?
inputs??????????
p 

 
? "???????????
"__inference_signature_wrapper_2395?"0123456789:;<=>?@ABCDEFGHIJKLMNOPQE?B
? 
;?8
6
input_2+?(
input_2???????????"7?4
2

sequential$?!

sequential??????????
?__inference_vgg16_layer_call_and_return_conditional_losses_1611?0123456789:;<=>?@ABCDEFGHIB??
8?5
+?(
input_1???????????
p

 
? "&?#
?
0??????????
? ?
?__inference_vgg16_layer_call_and_return_conditional_losses_1660?0123456789:;<=>?@ABCDEFGHIB??
8?5
+?(
input_1???????????
p 

 
? "&?#
?
0??????????
? ?
?__inference_vgg16_layer_call_and_return_conditional_losses_2842?0123456789:;<=>?@ABCDEFGHIA?>
7?4
*?'
inputs???????????
p

 
? "&?#
?
0??????????
? ?
?__inference_vgg16_layer_call_and_return_conditional_losses_2944?0123456789:;<=>?@ABCDEFGHIA?>
7?4
*?'
inputs???????????
p 

 
? "&?#
?
0??????????
? ?
$__inference_vgg16_layer_call_fn_1741{0123456789:;<=>?@ABCDEFGHIB??
8?5
+?(
input_1???????????
p

 
? "????????????
$__inference_vgg16_layer_call_fn_1821{0123456789:;<=>?@ABCDEFGHIB??
8?5
+?(
input_1???????????
p 

 
? "????????????
$__inference_vgg16_layer_call_fn_2975z0123456789:;<=>?@ABCDEFGHIA?>
7?4
*?'
inputs???????????
p

 
? "????????????
$__inference_vgg16_layer_call_fn_3006z0123456789:;<=>?@ABCDEFGHIA?>
7?4
*?'
inputs???????????
p 

 
? "???????????