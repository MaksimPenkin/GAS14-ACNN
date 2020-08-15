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
            j = self.conv_blocks_num//2 - 1
            finish_block = self.conv_blocks_num//2 + 1

            self.H_conv[0] = conv2d(inputs, self.feature_size, self.cnn_size, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable)
            for i in range(self.conv_blocks_num):
                if (i > self.conv_blocks_num//2):
                    self.H_conv[finish_block] = ResBlock(self.H_conv[finish_block] + self.H_conv[j], self.feature_size, ksize=self.cnn_size, 
                        regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, scope='ResBlock{0}'.format(i))
                    j=j-1
                else:
                    self.H_conv[i+1] = ResBlock(self.H_conv[i], self.feature_size, ksize=self.cnn_size, 
                        regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable, scope='ResBlock{0}'.format(i))

            inp_pred = conv2d(self.H_conv[finish_block] + self.H_conv[0], self.chns, self.cnn_size, activation=None, regularizator=slim.l2_regularizer(scale=self.reg_scale), reuse=var_reuse, trainable=var_trainable) + inputs
            var_reuse=True

        return inp_pred
