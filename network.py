import numpy as np
import w_loss,c_loss,data_gen
import random
from keras.layers import Dense,Conv1D,Concatenate,Flatten,Lambda,Activation,Multiply,Add,GlobalMaxPooling1D,Dropout
from keras.callbacks import ModelCheckpoint,Callback
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras import backend as K
from keras.engine import Input, Model
from keras import regularizers,initializers
from tensorflow.python.ops import array_ops, nn

losses=['w_loss','c_loss','g_loss'] #available losses
chosen_loss=0#0 for the wasserstein loss, 1 for the N-pair cosine loss
use_quantile=True #true to aggregate using quantile, false for global max aggregation
classification=False #false for metric learning setting,true for a classification scenario
feature_no=128 #number of aggregated filters
quantile_no=16 #number of quantiles to represent the activations for each filter
checkpoint_save=10 #save the trained model every 10 epochs
batch_size=100
learning_rate=0.0001
epchs_no=2000
adam=Adam(learning_rate)

class act(Layer):
    def __init__(self,**kwargs):
        super(act, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.alpha= self.add_weight(name='{}_act'.format(self.name), shape=input_shape[0], initializer='zeros',regularizer=regularizers.l2(1), trainable=True)
    
    def compute_output_shape(self,input_shape):
        return (input_shape)
    
    def call(self, x, mask=None):
        return K.tf.nn.relu(x)-(1-self.alpha)*K.tf.nn.relu(-x)

class quantile(Layer):
    def __init__(self,qu_no, ft_no,**kwargs):
        self.no=qu_no
        self.ft=ft_no
        self.quan_init=np.array([np.append(-np.log((np.float(qu_no)/np.array(range(1,qu_no)))-1),10)]*self.ft)
        super(quantile, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.quan = K.variable(self.quan_init, name='{}_quantile'.format(self.name))
        self.trainable_weights=[self.quan]
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],self.no)
    
    def call(self, x, mask=None):
        bs=K.shape(x)[0]
        l=K.tf.cast(x[:,0,-1],dtype='float32')
        index_all=K.tf.multiply(K.tf.tile(K.tf.expand_dims((1-K.tf.sigmoid(self.quan)),0),[bs,1,1]),(l[:,K.tf.newaxis,K.tf.newaxis]-1))
        index_floor_cal=K.tf.concat([K.tf.stack([K.tf.transpose(K.tf.tile([K.tf.transpose(K.tf.tile([K.tf.range(bs)],[self.no,1]))],[self.ft,1,1]),perm=[1,0,2]),K.tf.tile([K.tf.transpose(K.tf.tile([K.tf.range(self.ft)],[self.no,1]))],[bs,1,1])],3),K.tf.expand_dims(K.tf.cast(K.tf.floor(index_all),dtype='int32'),axis=3)],axis=3)
        index_ceil_cal=K.tf.concat([K.tf.stack([K.tf.transpose(K.tf.tile([K.tf.transpose(K.tf.tile([K.tf.range(bs)],[self.no,1]))],[self.ft,1,1]),perm=[1,0,2]),K.tf.tile([K.tf.transpose(K.tf.tile([K.tf.range(self.ft)],[self.no,1]))],[bs,1,1])],3),K.tf.expand_dims(K.tf.cast(K.tf.ceil(index_all),dtype='int32'),axis=3)],axis=3)    
        y_1=K.tf.gather_nd(x,index_floor_cal)
        y_2=K.tf.gather_nd(x,index_ceil_cal)
        return (y_1 +K.tf.multiply((index_all-K.tf.floor(index_all)),(y_2-y_1)))

def n_b(l_in):
	for itr in range(16):
		l_in = Conv1D(2**(4+itr/4),3,kernel_initializer= initializers.glorot_uniform(seed=1))(l_in)
		l_in=act()(l_in)
	return l_in


feature_input=Input(shape=(None,6,))
mask_input=Input(shape=(None,feature_no,))
len_input=Input(shape=(feature_no,1))
cnv=n_b(feature_input)
cnv_mask=Add()([cnv,mask_input])
if(use_quantile):
    lmd=Lambda(lambda x:K.tf.nn.top_k(K.tf.matrix_transpose(x),k=array_ops.shape(x)[1]).values)(cnv_mask)
    len_concat=Concatenate()([lmd,len_input])
    qnt=quantile(quantile_no,feature_no)(len_concat)
    out=Flatten()(qnt)
else:
    out=GlobalMaxPooling1D()(cnv_mask)
    weights=[]
    chosen_loss=1
if(classification):
    drp=Dropout(0.5,seed=1)(out)
    out=Dense(105, activation='softmax',kernel_initializer=initializers.glorot_uniform(seed=1))(drp)
model=Model(inputs=[feature_input,mask_input,len_input], output=out)
if(use_quantile):
    weights=model.layers[38].weights[0]
if(classification):
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
else:
    model.compile(loss=globals()[losses[chosen_loss]].loss_emb(weights,feature_no,quantile_no,batch_size), optimizer=adam)
x_red,y,train_x_id,test_x_id,valid_x_id,x_mask,x_exp_pad=data_gen.load_data()
max_len=np.max(x_exp_pad)

def dataGenerator_mask(curr_x_ids):
    random.seed(1)
    individual_seq_no=2
    batch_individual_no=(batch_size/2)
    y_current=y[curr_x_ids]
    x_red_current=np.array(x_red)[curr_x_ids]
    x_mask_current=np.array(x_mask)[curr_x_ids]
    x_len_current=max_len-x_exp_pad[curr_x_ids]-((3-1)*quantile_no)
    all_ids=np.unique(y_current)
    while True:
        selected_ids=all_ids[np.random.choice(len(all_ids),batch_individual_no,replace=False)]
        individual_sel_seq_agg=[]
        ycat_agg=[]
        for ij in selected_ids:
            individual_seq=np.where(y_current==ij)[0]
            individual_sel_seq=individual_seq[np.random.choice(len(individual_seq),individual_seq_no,replace=False)]
            individual_sel_seq_agg.extend(individual_sel_seq)
        yield [[np.array(x_red_current[individual_sel_seq_agg]),np.array(x_mask_current[individual_sel_seq_agg]),np.expand_dims(np.transpose(np.array([x_len_current[individual_sel_seq_agg]]*feature_no)),axis=-1)],np.array(y_current[individual_sel_seq_agg])]

mc= keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5',save_weights_only=True,period=10)
model.fit_generator(generator=dataGenerator_mask(train_x_id),epochs=epchs_no, steps_per_epoch=len(train_x_id)/(batch_size/2),callbacks=[mc])
