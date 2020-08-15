""" 
 @author   Maksim Penkin
https://github.com/MaksimPenkin

"""

    def generator(self, inputs, mode='train', reuse=False, scope='g_net'):
        var_reuse = reuse
        var_trainable=False
        if (mode=='train'):
            var_trainable=True
        with tf.compat.v1.variable_scope(scope):
            conv_pre = conv2d(inputs, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable)

            conv1 = residual_channel_attention_block(conv_pre, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN1')
            conv2 = residual_channel_attention_block(conv1, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN2')
            conv3 = residual_channel_attention_block(conv2, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN3')
            
            conv4 = residual_channel_attention_block(conv3, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN4') + conv2
            conv5 = residual_channel_attention_block(conv4, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN5') + conv1
            conv6 = residual_channel_attention_block(conv5, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, is_training=self.is_training, scope='RCAN6') + conv_pre

            conv_restore = conv2d(conv6, 16, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable)
            inp_pred = conv2d(conv_restore, self.chns, self.cnn_size, activation=None, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable) + inputs
            var_reuse=True
        
        return inp_pred