import tensorflow as tf

@tf.function
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    '''Dice coefficient for binary segmentation'''
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

@tf.function
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

@tf.function
def combined_loss(y_true, y_pred):
    '''Combined loss function that combines Dice loss with binary cross-entropy'''
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice + bce

@tf.function
def uncertainty_aware_loss(y_true, y_pred, lambda_=0.01):
    '''Uncertainty-aware loss function that combines Dice loss with uncertaint
    y'''
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    uncertainty = tf.abs(tf.math.reduce_std(y_pred))  

    # Combine the base losses and multiply by the uncertainty factor
    base_loss = bce + dice
    weighted_loss = base_loss * (1 + lambda_ * uncertainty)

    return tf.reduce_mean(weighted_loss)


#################### Bayesian U-Net Loss Functions ####################
@tf.function
def bayesian_unet_nll(y_true, y_pred):
    '''
    Negative log-likelihood for Bayesian U-Net.
    it works with probabilistic layers from TensorFlow Probability, we add log_prop to the loss function,
    because the output of the model is a distribution.
    '''
    return -tf.reduce_mean(y_pred.log_prob(y_true))

@tf.function
def combined_loss_bayesian_unet(y_true, y_pred):
    """Combine the Dice loss with negative log-likelihood for probabilistic layers."""
    # print("y_pred distribution mean shape:", y_pred.mean().shape)
    dice_loss_value = dice_loss(y_true, y_pred.mean())  # .mean() to convert distribution to its expected value
    nll_loss = -tf.reduce_mean(y_pred.log_prob(y_true))  # Negative log likelihood
    return dice_loss_value + nll_loss
    # y_pred_mean = y_pred.mean()
    # if len(y_pred_mean.shape) > len(y_true.shape):
    #     y_pred_mean = tf.squeeze(y_pred_mean, axis=-1)  # Squeeze out the last dimension if it's extra
    # dice_loss_value = dice_loss(y_true, y_pred_mean)
    # nll_loss = -tf.reduce_mean(y_pred.log_prob(y_true))
    # return dice_loss_value + nll_loss