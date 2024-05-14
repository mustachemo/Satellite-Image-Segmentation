from tensorflow.keras import layers
from tensorflow.keras.models import Model


def downsample_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x, layers.MaxPooling2D(pool_size=(2, 2))(x)

def double_conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)

def upsample_block(input_tensor, skip_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = layers.concatenate([x, skip_tensor])
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)


def build_unet_model():
    # inputs
    inputs = layers.Input(shape=(256, 256, 3))

    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    bottleneck = double_conv_block(p4, 1024)

    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    outputs = layers.Conv2D(1, (1, 1), padding="same", activation = "sigmoid")(u9)

    unet_model = Model(inputs, outputs, name="U-Net")
    return unet_model


# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
# from tensorflow.keras.models import Model


# def build_unet(input_size=(256, 256, 3), dropout_rate=0.5):
#     inputs = Input(input_size)
    
#     # Downsample
#     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
#     c1 = BatchNormalization()(c1)
#     c1 = Dropout(dropout_rate)(c1)
#     c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
#     p1 = MaxPooling2D((2, 2))(c1)
    
#     # Bottleneck
#     c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p1)
#     c5 = Dropout(dropout_rate)(c5)
#     c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    
#     # Upsample
#     u6 = UpSampling2D((2, 2))(c5)
#     u6 = concatenate([u6, c1])
#     c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
#     c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
    
#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(c6)
#     model = Model(inputs, outputs)
#     return model
