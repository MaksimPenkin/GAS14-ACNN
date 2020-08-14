""" 
 @author   Maxim Penkin
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

            weights_gen = conv2d(inputs, 64, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, activation=None, trainable=var_trainable)
            weights_gen = tf.nn.relu(tf.compat.v1.layers.batch_normalization(weights_gen, training=self.is_training, trainable=var_trainable, reuse=var_reuse))
            weights_gen = conv2d(weights_gen, 128, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), activation=tf.nn.relu, reuse=var_reuse, trainable=var_trainable)
            weights = conv2d(weights_gen, 256, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), activation=tf.nn.leaky_relu, reuse=var_reuse, trainable=var_trainable)
            
            features = conv2d(conv6, 32, 1, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable)

            g_RNN1_x1 = weights[:, :, :,   0: 32]
            g_RNN1_y1 = weights[:, :, :,  32: 64]
            g_RNN1_x2 = weights[:, :, :,  64: 96]
            g_RNN1_y2 = weights[:, :, :,  96: 128]

            g_RNN2_x1 = weights[:, :, :,  128: 160]
            g_RNN2_y1 = weights[:, :, :,  160: 192]
            g_RNN2_x2 = weights[:, :, :,  192: 224]
            g_RNN2_y2 = weights[:, :, :,  224: 256]

            rnn1_1 = lrnn(features, g_RNN1_x1, horizontal=True,  reverse=False)
            rnn1_2 = lrnn(features, g_RNN1_x2, horizontal=True,  reverse=True)
            rnn1_3 = lrnn(features, g_RNN1_y1, horizontal=False, reverse=False)
            rnn1_4 = lrnn(features, g_RNN1_y2, horizontal=False, reverse=True)
            rnn1 = tf.concat([rnn1_1, rnn1_2, rnn1_3, rnn1_4], axis=-1)
            conv_rnn1 = conv2d(rnn1, 32, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable)

            rnn2_1 = lrnn(conv_rnn1, g_RNN2_x1, horizontal=True,  reverse=False)
            rnn2_2 = lrnn(conv_rnn1, g_RNN2_x2, horizontal=True,  reverse=True)
            rnn2_3 = lrnn(conv_rnn1, g_RNN2_y1, horizontal=False, reverse=False)
            rnn2_4 = lrnn(conv_rnn1, g_RNN2_y2, horizontal=False, reverse=True)
            rnn2 = tf.concat([rnn2_1, rnn2_2, rnn2_3, rnn2_4], axis=-1)
            conv_rnn2 = conv2d(rnn2, 32, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable)+features

            conv_restore = conv2d(conv_rnn2, 16, 3, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable)
            inp_pred = conv2d(conv_restore, self.chns, self.cnn_size, activation=None, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable) + inputs
            var_reuse=True
        
        return inp_pred