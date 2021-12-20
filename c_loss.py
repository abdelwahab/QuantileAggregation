from tensorflow.python.ops import array_ops, math_ops, nn
from keras import backend as K
import numpy as np


def loss_emb(quantiles,feature_no,quantile_no,batch_size):
    def loss_split(yTrue,yPred):
        return pair_loss(K.tf.boolean_mask(yTrue,np.array([True,False]*(batch_size/2))),K.tf.boolean_mask(yPred,np.array([True,False]*(batch_size/2))),K.tf.boolean_mask(yPred,np.array([False,True]*(batch_size/2))))
    
    def npairs_loss(labels, embeddings_anchor, embeddings_positive):
        reg_anchor = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
        reg_positive = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
        l2loss = math_ops.multiply(0.0005, reg_anchor + reg_positive, name='l2loss')
        similarity_matrix = math_ops.matmul(embeddings_anchor, embeddings_positive, transpose_a=False,transpose_b=True)
        lshape = array_ops.shape(labels)
        labels = array_ops.reshape(labels, [lshape[0], 1])
        labels_remapped = math_ops.to_float(math_ops.equal(labels, array_ops.transpose(labels)))
        labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)
        xent_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=K.tf.cast(range(0,(batch_size/2)),dtype='int32'))
        xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')
        return l2loss + xent_loss
    
    return loss_split