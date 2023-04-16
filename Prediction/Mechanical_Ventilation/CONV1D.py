import tensorflow.compat.v1 as tf
from sklearn.utils.class_weight import compute_class_weight
tf.disable_v2_behavior()
from tensorflow.compat.v1  import set_random_seed
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, RepeatVector, Lambda
from keras.layers import Input, Conv2D, Conv1D, Conv3D, MaxPooling2D, MaxPooling1D
from keras.layers import Concatenate
from keras import backend as K
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import random as rn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

BATCH_SIZE = 128
EPOCHS = 12
DROPOUT = 0.5
RANDOM = 0
NUM_CLASSES = 4

def CONV1D_Model(x_train, y_train, y_train_classes, x_val, y_val_classes, x_test, y_test_classes):
    class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(zip(range(len(class_weight)), class_weight))


    sess = tf.Session(graph=tf.get_default_graph())
    K.set_session(sess)

    np.random.seed(RANDOM)
    set_random_seed(RANDOM)
    rn.seed(RANDOM)

    input_shape = (x_train.shape[1], x_train.shape[2])
    inputs = Input(shape=input_shape)
    model = Conv1D(64, kernel_size=3,
                 strides=1,
                 activation='relu',
                 input_shape=input_shape,
                 padding='same',
                 name='conv2')(inputs)

    model = (MaxPooling1D(pool_size=3, strides=1))(model)

    model2 = Conv1D(64, kernel_size=4,
                 strides=1,
                 activation='relu',
                 input_shape=input_shape,
                 padding='same',
                 name='conv3')(inputs)

    model2 = MaxPooling1D(pool_size=3, strides=1)(model2)

    model3 = Conv1D(64, kernel_size=5,
                 strides=1,
                 activation='relu',
                 input_shape=input_shape,
                 padding='same',
                 name='conv4')(inputs)

    model3 = MaxPooling1D(pool_size=3, strides=1)(model3)

    models = [model, model2, model3]

    full_model = keras.layers.concatenate(models)
    full_model = Flatten()(full_model)
    full_model = Dense(128, activation='relu')(full_model)
    full_model = Dropout(DROPOUT)(full_model)
    full_model = Dense(NUM_CLASSES, activation='softmax')(full_model)

    full_model = keras.models.Model(inputs, full_model)

    # The actual training is done here
    full_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=.0005),
              metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    full_model.fit(x_train, y_train_classes,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          class_weight=class_weight,
          callbacks=[early_stopping],
          validation_data=(x_val, y_val_classes))



    full_model.save('../../mymodel.h5')
    full_model.summary()

    print("Saved model to disk")


    test_preds_cnn = full_model.predict(x_test, batch_size=BATCH_SIZE)
        #test_preds_proba_cnn = full_model.predict_proba(x_test, batch_size=BATCH_SIZE)
    #print('predict_results', test_preds_cnn)
    #test_preds_proba_cnn = full_model.predict_proba(x_test, batch_size=BATCH_SIZE)
    #print('predict_proba_results', test_preds_proba_cnn)
    idx = np.argmax(test_preds_cnn, axis=-1)
    test_preds_cnn_label = np.zeros(test_preds_cnn.shape)
    test_preds_cnn_label[np.arange(test_preds_cnn_label.shape[0]), idx] = 1

    print("AUC:")
    print(roc_auc_score(y_test_classes, test_preds_cnn, average=None))
    print("AUC Macro:")
    print(roc_auc_score(y_test_classes, test_preds_cnn, average='macro'))
    print("Accuracy: ")
    print(accuracy_score(y_test_classes, test_preds_cnn_label))
    print("F1 Macro:")
    print(f1_score(y_test_classes, test_preds_cnn_label, average='macro'))
    print("AUPRC Macro: ")
    print(average_precision_score(y_test_classes, test_preds_cnn_label, average='macro'))
