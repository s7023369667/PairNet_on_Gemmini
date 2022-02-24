import keras.backend as K


def OaP_GD_loss(y_true, y_pred):
    qq = 0.00001
    LF = K.switch(K.equal(y_true, 1),
                  lambda: -1 * K.pow((1 - y_pred), 2) * K.log(y_pred + qq),
                  lambda: -1 * K.pow((1 - y_true), 4) * K.pow(y_pred, 2) * K.log(1 - y_pred + qq))
    return LF
    # y_true = 1  K.pow( (1-y_pred) ,2) * K.log(y_pred)
    # else        K.pow( (1-y_true) ,4) * K.pow(y_pred,2) * K.log(1-y_pred)