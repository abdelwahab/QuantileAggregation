from tensorflow.python.ops import array_ops, math_ops, nn
from keras import backend as K
import numpy as 

def loss_gaussian():
    def features(embeddings_positive, embeddings_anchor):
        embeddings_positive_m=K.mean(embeddings_positive,1)
        embeddings_anchor_m=K.mean(embeddings_anchor,1)
        embeddings_positive_std=K.relu(K.var(embeddings_positive,1))
        embeddings_anchor_std=K.relu(K.var(embeddings_anchor,1))
        return embeddings_positive_m,embeddings_anchor_m,embeddings_positive_std,embeddings_anchor_std

    def distribution_distance(embeddings_positive, embeddings_anchor):
        embeddings_positive_m, embeddings_anchor_m, embeddings_positive_std, embeddings_anchor_std=features(embeddings_positive, embeddings_anchor)
        sigma_sq_fuse = K.relu(K.tf.transpose(embeddings_positive_std[None,:,:] + embeddings_anchor_std[:,None,:],(1,0,2)))
        diffs = K.tf.pow(K.tf.transpose(embeddings_positive_m[None,:,:] - embeddings_anchor_m[:,None,:],(1,0,2)),2)/(1e-10 + sigma_sq_fuse) + K.tf.log(1e-10 + sigma_sq_fuse)
        return -1*K.tf.reduce_sum(diffs, axis=2)

    def loss_split(yTrue, yPred):
        return npairs_loss(K.tf.boolean_mask(yTrue, np.array([True, False] * int(batch_size/2))),
                           K.tf.boolean_mask(yPred, np.array([True, False] * int(batch_size/2))),
                           K.tf.boolean_mask(yPred, np.array([False, True] * int(batch_size/2)))) 

    def npairs_loss(labels, embeddings_anchor, embeddings_positive, reg_lambda=0.002):
        reg_anchor = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_anchor), 1))
        reg_positive = math_ops.reduce_mean(math_ops.reduce_sum(math_ops.square(embeddings_positive), 1))
        l2loss = math_ops.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')
        embeddings_anchor = K.tf.reshape(embeddings_anchor, [int(batch_size/2), 768, 128])
        embeddings_positive = K.tf.reshape(embeddings_positive, [int(batch_size/2), 768, 128])
        similarity_matrix = distribution_distance(embeddings_positive, embeddings_anchor)
        lshape = array_ops.shape(labels)
        labels = array_ops.reshape(labels, [lshape[0], 1])
        labels_remapped = math_ops.to_float(math_ops.equal(labels, array_ops.transpose(labels)))
        labels_remapped /= math_ops.reduce_sum(labels_remapped, 1, keep_dims=True)
        xent_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=K.tf.cast(range(0, 50),dtype='int32')) 
        xent_loss = math_ops.reduce_mean(xent_loss, name='xentropy')
        return l2loss + xent_loss

    return loss_split
