from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint
from models.utils_je import metric
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
        l.trainable = True

    bert_model_2 = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, seq_len=None)
    for l in bert_model_2.layers:
        l.trainable = True

    tokens_in = Input(shape=(None,))
    segments_in = Input(shape=(None,))

    sub_type_in = Input(shape=(None,))
    sub_head_idx_in = Input(shape=(1,))
    sub_tail_idx_in = Input(shape=(1,))

    sub_tokens_in = Input(shape=(None,))
    sub_segments_in = Input(shape=(None,))

    gold_obj_heads_in = Input(shape=(None, num_rels))
    gold_obj_tails_in = Input(shape=(None, num_rels))

    tokens, segments = tokens_in, segments_in
    sub_type, sub_head_idx, sub_tail_idx = sub_type_in, sub_head_idx_in, sub_tail_idx_in
    sub_tokens, sub_segments = sub_tokens_in, sub_segments_in
    gold_obj_heads, gold_obj_tails = gold_obj_heads_in, gold_obj_tails_in

    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(tokens)

    tokens_feature = bert_model([tokens, segments])
    seq_feature = Lambda(lambda tokens_feature: tokens_feature[:, 0])(tokens_feature)

    pred_sub_types = Dense(3, activation='softmax')(seq_feature)
    subject_model = Model([tokens_in, segments_in], pred_sub_types)

    tokens_feature = bert_model_2([tokens, segments])

    tokens_feature = Add()([tokens_feature, seq_feature])

    pred_obj_heads = Dense(num_rels, activation='sigmoid')(tokens_feature)
    pred_obj_tails = Dense(num_rels, activation='sigmoid')(tokens_feature)

    object_model = Model(
        [tokens_in, segments_in, sub_type_in, sub_head_idx_in, sub_tail_idx_in, sub_tokens_in, sub_segments_in],
        [pred_obj_heads, pred_obj_tails])

    ebm_model = Model(
        [tokens_in, segments_in, sub_type_in, sub_head_idx_in, sub_tail_idx_in, sub_tokens_in, sub_segments_in,
         gold_obj_heads_in,
         gold_obj_tails_in],
        [pred_sub_types, pred_obj_heads, pred_obj_tails])

    sub_type_loss = K.categorical_crossentropy(sub_type, pred_sub_types)

    obj_heads_loss = K.sum(K.binary_crossentropy(gold_obj_heads, pred_obj_heads), keepdims=False)
    obj_tails_loss = K.sum(K.binary_crossentropy(gold_obj_tails, pred_obj_tails), keepdims=False)

    a_sub = K.stop_gradient(sub_type_loss)
    a_obj = K.stop_gradient(obj_heads_loss + obj_tails_loss)

    loss = sub_type_loss / a_sub + (obj_heads_loss + obj_tails_loss) / a_obj
    ebm_model.add_loss(loss)
    ebm_model.compile(optimizer=Adam(LR))
    ebm_model.summary()

    return subject_model, object_model, ebm_model


class Evaluate(Callback):
    def __init__(self, subject_model, object_model, tokenizer, id2rel, eval_data, sub2text, sub_labels,
                 save_weights_path, min_delta=1e-4,
                 patience=10):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater
        self.subject_model = subject_model
        self.object_model = object_model
        self.tokenizer = tokenizer
        self.id2rel = id2rel
        self.eval_data = eval_data
        self.sub2text = sub2text
        self.sub_labels = sub_labels
        self.save_weights_path = save_weights_path

    def on_train_begin(self, logs=None):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.warmup_epochs = 2
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        precision, recall, f1 = metric(self.subject_model, self.object_model, self.eval_data, self.id2rel,
                                       self.tokenizer, self.sub2text, self.sub_labels)
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
