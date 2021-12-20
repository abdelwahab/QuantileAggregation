from tensorflow.python.ops import array_ops, math_ops, nn
from keras import backend as K
import numpy as np

def loss_emb(quantiles,feature_no,quantile_no,batch_size):
    q=K.tf.concat([K.tf.sigmoid(quantiles),K.tf.expand_dims([1.0]*feature_no,1)],axis=1)
    q=-q
    sortedv=K.tf.nn.top_k(q,k=array_ops.shape(q)[1])
    sortedi=sortedv.indices
    ind_sort=K.tf.concat([K.tf.reshape(K.tf.tile(K.tf.reshape(K.tf.constant(range(q.shape[0]),dtype='int32'),[-1,1]),[1,q.shape[-1]]),[q.shape[0],q.shape[-1],1]),K.tf.expand_dims(sortedi,-1)],-1)
    bt_sz=(batch_size/2)*(batch_size/2)
    ind_sort_batch=K.tf.concat([K.tf.expand_dims(K.reshape(K.tf.tile(K.tf.reshape(K.tf.constant(range(bt_sz),dtype='int32'),[-1,1]),[1,ind_sort.shape[0]*ind_sort.shape[1]]),[bt_sz,ind_sort.shape[0],ind_sort.shape[1]]),axis=-1), K.tf.tile(K.tf.expand_dims(ind_sort,0),[bt_sz,1,1,1])],axis=-1)
    q=-sortedv.values
    qstep=q[:,1:]-q[:,:-1]
    eps=1e-10
    def distribution_distance(embeddings_positive,embeddings_anchor):
        data_x=K.tf.reshape(K.tf.concat([(embeddings_positive[None,:,None,:,:]+K.tf.zeros_like(embeddings_anchor[:,None,None,:,:])),(embeddings_anchor[:,None,None,:,:]+K.tf.zeros_like(embeddings_positive[None,:,None,:,:]))],axis=2),[-1,2,128,17])
        v1=data_x[:,0]
        v1=K.tf.gather_nd(v1,ind_sort_batch)
        v2=data_x[:,1]
        v2=K.tf.gather_nd(v2,ind_sort_batch)
        v1step=v1[:,:,1:]-v1[:,:,:-1]
        v2step=v2[:,:,1:]-v2[:,:,:-1]
        m1=(qstep)/(K.tf.abs(v1step)+eps)*K.tf.sign(v1step)
        m2=(qstep)/(K.tf.abs(v2step)+eps)*K.tf.sign(v2step)
        vs=((m1*v1[:,:,:-1]-m2*v2[:,:,:-1])/(K.tf.abs(m1-m2)+eps))*K.tf.sign(m1-m2)
        qs=((((v2[:,:,:-1]-v1[:,:,:-1])*m1*m2)/(K.tf.abs(m2-m1)+eps))*K.tf.sign(m2-m1))+q[:,:-1]
        p1=K.tf.maximum(0.0,qs-q[:,:-1])
        p2=K.tf.maximum(0.0,q[:,1:]-qs)
        ppsign=K.tf.sign(p1*p2)
        v2v1step=K.tf.abs(v2-v1)
        A1=v2v1step[:,:,:-1]*p1*0.5*ppsign
        A2=v2v1step[:,:,1:]*p2*0.5*ppsign
        A3=(v2v1step[:,:,1:]+v2v1step[:,:,:-1])*(qstep)*0.5*(1-ppsign)
        return K.tf.reduce_sum(K.tf.reduce_sum(A1+A2+A3,axis=-1),axis=-1)
    
    def loss_split(yTrue,yPred):
        return pair_loss(K.tf.boolean_mask(yTrue,np.array([True,False]*(batch_size/2))),K.tf.boolean_mask(yPred,np.array([True,False]*(batch_size/2))),K.tf.boolean_mask(yPred,np.array([False,True]*(batch_size/2))))
    
    def pair_loss(labels, embeddings_anchor, embeddings_positive):
        reg_anchor = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
        reg_positive = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
        l2loss = math_ops.multiply(0.0005, reg_anchor + reg_positive, name='l2loss')
        embeddings_anchor=K.tf.reshape(embeddings_anchor,[(batch_size/2),feature_no,quantile_no])
        embeddings_anchor=K.tf.concat([embeddings_anchor,K.tf.expand_dims(embeddings_anchor[:,:,-1]+1e-5,2)],axis=2)
        embeddings_positive=K.tf.reshape(embeddings_positive,[(batch_size/2),feature_no,quantile_no])
        embeddings_positive=K.tf.concat([embeddings_positive,K.tf.expand_dims(embeddings_positive[:,:,-1]+1e-5,2)],axis=2)
        similarity_matrix = -K.tf.reshape(distribution_distance(embeddings_positive,embeddings_anchor),[(batch_size/2),(batch_size/2)])
        lshape = array_ops.shape(labels)
        labels = array_ops.reshape(labels, [lshape[0], 1])
        labels_remapped = math_ops.to_float(math_ops.equal(labels, array_ops.transpose(labels)))
        labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)
        xent_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=K.tf.cast(range(0,(batch_size/2)),dtype='int32'))
        xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')
        return l2loss + xent_loss
    return loss_split