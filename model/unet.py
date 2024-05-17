from tensorflow.keras import layers
from tensorflow.keras.models import Model
from configs import X_DIMENSION, Y_DIMENSION, ACTIVATION_FUNC

def downsample_block(input_tensor, num_filters, dropout_rate=0.1):
    """Block for downsampling: Convolution -> Batch Normalization -> ReLU -> Convolution -> Batch Normalization -> ReLU -> Max Pooling"""
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x, layers.MaxPooling2D(pool_size=(2, 2))(x)

def double_conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation(ACTIVATION_FUNC)(x)

def upsample_block(input_tensor, skip_tensor, num_filters, dropout_rate=0.1):
    """Block for upsampling: Transpose Convolution -> Concatenation with skip connection -> Convolution -> Batch Normalization -> ReLU -> Convolution -> Batch Normalization -> ReLU"""
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = layers.concatenate([x, skip_tensor])
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x


def build_unet_model(dropout_rate=0.1):
    """Build U-Net model incorporating dropout for uncertainty estimation."""
    inputs = layers.Input(shape=(Y_DIMENSION, X_DIMENSION, 3))

    # Encoding path
    f1, p1 = downsample_block(inputs, 64, dropout_rate)
    f2, p2 = downsample_block(p1, 128, dropout_rate)
    f3, p3 = downsample_block(p2, 256, dropout_rate)
    f4, p4 = downsample_block(p3, 512, dropout_rate)

    # Bottleneck
    bottleneck = double_conv_block(p4, 1024)
    bottleneck = layers.SpatialDropout2D(dropout_rate)(bottleneck)  # Additional dropout at the bottleneck
    
    # Decoding path
    u6 = upsample_block(bottleneck, f4, 512, dropout_rate)
    u7 = upsample_block(u6, f3, 256, dropout_rate)
    u8 = upsample_block(u7, f2, 128, dropout_rate)
    u9 = upsample_block(u8, f1, 64, dropout_rate)


    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u9)

    unet_model = Model(inputs, outputs, name="U-Net")
    return unet_model
