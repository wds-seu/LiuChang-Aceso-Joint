from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint
from models.utils_se import metric
from models.attention import AttentionLayer
import numpy as np


class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


def E2EModel(bert_config_path, bert_checkpoint_path, LR, num_rels):
    bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = False

    tokens_in = Input(shape=(None,))
    segments_in = Input(shape=(None,))
    gold_sub_type_in = Input(shape=(None,))

    tokens, segments, gold_sub_type = tokens_in, segments_in, gold_sub_type_in

    tokens_feature = bert_model([tokens, segments])

    tokens_feature = NonMasking()(tokens_feature)
    seq_feature = Bidirectional(GRU(64, dropout=0.2, return_sequences=True))(tokens_feature)
    seq_feature = AttentionLayer(attention_size=50)(seq_feature)

    pred_sub_types = Dense(4, activation='softmax')(seq_feature)

    subject_model = Model([tokens_in, segments_in], pred_sub_types)
    sub_model = Model([tokens_in, segments_in, gold_sub_type_in], pred_sub_types)

    e = 0.2
    sub_type_loss_1 = K.categorical_crossentropy(gold_sub_type, pred_sub_types)
    sub_type_loss_2 = K.categorical_crossentropy(K.ones_like(pred_sub_types) / 4, pred_sub_types)
    loss = (1 - e) * sub_type_loss_1 + e * sub_type_loss_2

    sub_model.add_loss(loss)
    sub_model.compile(optimizer=Adam(LR))
    sub_model.summary()

    return sub_model, subject_model


class Evaluate(Callback):
    def __init__(self, subject_model, tokenizer, id2rel, eval_data, save_weights_path, min_delta=1e-4, patience=7):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater
        self.subject_model = subject_model

        self.tokenizer = tokenizer
        self.id2rel = id2rel
        self.eval_data = eval_data
        self.save_weights_path = save_weights_path

    def on_train_begin(self, logs=None):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.warmup_epochs = 2
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        precision, recall, f1 = metric(self.subject_model, self.eval_data, self.id2rel, self.tokenizer)
        if self.monitor_op(f1 - self.min_delta, self.best) or self.monitor_op(self.min_delta, f1):
            self.best = f1
            self.wait = 0
            self.model.save_weights(self.save_weights_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
