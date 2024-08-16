import tensorflow as tf


class Level(tf.keras.layers.Layer):
    def __init__(self, downscale=1, vis_size=(240, 320), filters=256, name="level"):
        super().__init__(name=name)
        self.vis_size = vis_size
        self.filters = filters

        self.resize_down = tf.keras.layers.Resizing(
            vis_size[0] // downscale,
            vis_size[1] // downscale,
            interpolation='bilinear')
        self.resize_up = tf.keras.layers.Resizing(
            vis_size[0],
            vis_size[1],
            interpolation='bilinear')
        self.pre_conv = tf.keras.layers.Conv2D(filters=filters,
                                               kernel_size=1,
                                               use_bias=False)
        self.post_conv = tf.keras.layers.Conv2D(filters=filters,
                                                kernel_size=1,
                                                use_bias=False)

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        clip_x = inputs[0]
        vis = inputs[1]

        clip_x = self.resize_down(self.pre_conv(clip_x))
        vis = self.resize_down(vis)
        x = tf.concat([clip_x, vis], axis=-1)
        x = self.resize_up(self.post_conv(x))
        return x


class CLIPFeatureExtraction(tf.keras.layers.Layer):
    def __init__(self, shape=(240, 320, 256), name="clip_feature_extraction"):
        super().__init__(name=name)
        assert shape[2] == 256, f"Expected 256 input channels but got {shape[3]} which does not lead to 1024 output channels."
        pool_size = (shape[0] // 2, shape[1] // 2)

        # 4 features per feature map
        self.max_pool = tf.keras.layers.MaxPool2D(
            pool_size=pool_size, strides=pool_size, padding='valid')
        self.flatten = tf.keras.layers.Flatten()

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        return self.flatten(self.max_pool(inputs))


class CombineCLIPVisualV2(tf.keras.Model):
    def __init__(self, name="combine_clip_visual"):
        super().__init__(name=name)

        vis_size = (240, 320)
        filters = 256
        self.level_1 = Level(downscale=1, vis_size=vis_size, filters=filters)
        self.level_2 = Level(downscale=2, vis_size=vis_size, filters=filters)
        self.level_3 = Level(downscale=4, vis_size=vis_size, filters=filters)
        self.level_4 = Level(downscale=8, vis_size=vis_size, filters=filters)
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=1,
                                           use_bias=False)

        self.clip_feature_extraction = CLIPFeatureExtraction()
        self.clip_regulizer_loss = tf.keras.losses.CategoricalCrossentropy()
        self.up = tf.keras.layers.UpSampling2D(
            size=2, interpolation='bilinear')

    @tf.function(input_signature=[((tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="clip_features"),
                                   tf.TensorSpec(
                                       shape=(None, 56, 56, 256), dtype=tf.float32, name="clip_layer_1"),
                                   tf.TensorSpec(
                                       shape=(None, 28, 28, 512), dtype=tf.float32, name="clip_layer_2"),
                                   tf.TensorSpec(
                                       shape=(None, 14, 14, 1024), dtype=tf.float32, name="clip_layer_3"),
                                   tf.TensorSpec(
                                       shape=(None, 7, 7, 2048), dtype=tf.float32, name="clip_layer_4")),
                                   tf.TensorSpec(shape=(None, 240, 320, 256), dtype=tf.float32, name="visual_features"))])
    def call(self, inputs):
        clip_outputs = inputs[0]
        vis = inputs[1]                             # [(BN) 480 640 256]
        clip_features = clip_outputs[0]             # [(BN) 1024]
        clip_1 = clip_outputs[1]                    # [(BN) 56 56 256]
        clip_2 = clip_outputs[2]                    # [(BN) 28 28 512]
        clip_3 = clip_outputs[3]                    # [(BN) 14 14 1024]
        clip_4 = clip_outputs[4]                    # [(BN) 7 7 2048]

        x_level_1 = self.level_1((clip_1, vis))     # [(BN) 240 320 256]
        x_level_2 = self.level_2((clip_2, vis))     # [(BN) 240 320 256]
        x_level_3 = self.level_3((clip_3, vis))     # [(BN) 240 320 256]
        x_level_4 = self.level_4((clip_4, vis))     # [(BN) 240 320 256]

        x = tf.concat(                              # [(BN) 240 320 1024]
            [x_level_1, x_level_2, x_level_3, x_level_4], axis=-1)
        x = self.conv(x)                            # [(BN) 240 320 256]
        clip_features_pred = self.clip_feature_extraction(x)  # [(BN) 1024]
        loss = self.clip_regulizer_loss(clip_features, clip_features_pred)
        # Prevent the layer from ignoring the CLIP features and focus on vis
        self.add_loss(loss)
        x = self.up(x)
        return x


class CombineCLIPVisualV1(tf.keras.Model):
    def __init__(self, name="combine_clip_visual"):
        super().__init__(name=name)

        vis_size = (240, 320)
        filters = 256
        self.level_1 = Level(downscale=1, vis_size=vis_size, filters=filters)
        self.level_2 = Level(downscale=2, vis_size=vis_size, filters=filters)
        self.level_3 = Level(downscale=4, vis_size=vis_size, filters=filters)
        self.level_4 = Level(downscale=8, vis_size=vis_size, filters=filters)
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=1,
                                           use_bias=False)
        self.up = tf.keras.layers.UpSampling2D(
            size=2, interpolation='bilinear')

    @tf.function(input_signature=[((tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="clip_features"),
                                   tf.TensorSpec(
                                       shape=(None, 56, 56, 256), dtype=tf.float32, name="clip_layer_1"),
                                   tf.TensorSpec(
                                       shape=(None, 28, 28, 512), dtype=tf.float32, name="clip_layer_2"),
                                   tf.TensorSpec(
                                       shape=(None, 14, 14, 1024), dtype=tf.float32, name="clip_layer_3"),
                                   tf.TensorSpec(
                                       shape=(None, 7, 7, 2048), dtype=tf.float32, name="clip_layer_4")),
                                   tf.TensorSpec(shape=(None, 240, 320, 256), dtype=tf.float32, name="visual_features"))])
    def call(self, inputs):
        clip_outputs = inputs[0]
        vis = inputs[1]                             # [(BN) 480 640 256]
        _clip_features = clip_outputs[0]            # [(BN) 1024]
        clip_1 = clip_outputs[1]                    # [(BN) 56 56 256]
        clip_2 = clip_outputs[2]                    # [(BN) 28 28 512]
        clip_3 = clip_outputs[3]                    # [(BN) 14 14 1024]
        clip_4 = clip_outputs[4]                    # [(BN) 7 7 2048]

        x_level_1 = self.level_1((clip_1, vis))     # [(BN) 240 320 256]
        x_level_2 = self.level_2((clip_2, vis))     # [(BN) 240 320 256]
        x_level_3 = self.level_3((clip_3, vis))     # [(BN) 240 320 256]
        x_level_4 = self.level_4((clip_4, vis))     # [(BN) 240 320 256]

        x = tf.concat(                              # [(BN) 240 320 4*256]
            [x_level_1, x_level_2, x_level_3, x_level_4], axis=-1)
        x = self.conv(x)                            # [(BN) 240 320 256]
        x = self.up(x)                              # [(BN) 480 640 256]
        return x


class CombineCLIPVisualV0(tf.keras.Model):
    def __init__(self, name="combine_clip_visual_legacy"):
        super().__init__(name=name)

        filters = 256
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=1,
                                           use_bias=False)
        self.resize = tf.keras.layers.Resizing(
            240, 320, interpolation='bilinear')
        self.up = tf.keras.layers.UpSampling2D(
            size=2, interpolation='bilinear')

    @ tf.function(input_signature=[((tf.TensorSpec(shape=(None, 1024), dtype=tf.float32, name="clip_features"),
                                   tf.TensorSpec(
                                       shape=(None, 56, 56, 256), dtype=tf.float32, name="clip_layer_1"),
                                   tf.TensorSpec(
                                       shape=(None, 28, 28, 512), dtype=tf.float32, name="clip_layer_2"),
                                   tf.TensorSpec(
                                       shape=(None, 14, 14, 1024), dtype=tf.float32, name="clip_layer_3"),
                                   tf.TensorSpec(
                                       shape=(None, 7, 7, 2048), dtype=tf.float32, name="clip_layer_4")),
                                   tf.TensorSpec(shape=(None, 240, 320, 256), dtype=tf.float32, name="visual_features"))])
    def call(self, inputs):
        clip_outputs = inputs[0]
        visual_features = inputs[1]
        _clip_features = clip_outputs[0]                # [(BN) 1024]
        clip_256 = clip_outputs[1]                      # [(BN) 56 56 256]
        _clip_512 = clip_outputs[2]                     # [(BN) 28 28 512]
        _clip_1024 = clip_outputs[3]                    # [(BN) 14 14 1024]
        _clip_2048 = clip_outputs[4]                    # [(BN) 7 7 2048]

        clip_256r = self.resize(clip_256)               # [(BN) 240 320 256]
        fusion = tf.concat([clip_256r,                  # [(BN) 240 320 2*256]
                            visual_features], axis=-1)
        fusion = self.conv(fusion)                      # [(BN) 240 320 256]
        fusion = self.up(fusion)
        return fusion
