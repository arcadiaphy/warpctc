import cv2
import Queue
import random
import numpy as np
import mxnet as mx
import editdistance as ed
from multiprocessing import Process, Manager, Value
from plate.genplate import GenPlate, index, chars

from lstm import blstm_unroll


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class Prefetch:
    def __init__(self, count, worker, f):
        self.m = Manager()
        self.q = self.m.Queue(worker * 2)
        self.count = count
        self.worker = worker
        self.f = f
        self.stop = Value('b', False)
        self._produce()

    def _reset(self):
        if self.stop.value:
            self.stop.value = False
            self._produce()

    def _produce(self):
        def add_queue():
            random.seed()
            np.random.seed()
            while not self.stop.value:
                self.q.put(self.f())

        self.processes = [Process(target=add_queue) for _ in range(self.worker)]
        for p in self.processes:
            p.start()

    def _flush_queue(self):
        while 1:
            try:
                self.q.get_nowait()
            except Queue.Empty:
                break

    def _join(self):
        if not self.stop.value:
            self.stop.value = True
            self._flush_queue()  # ensure all processes not blocked

            for p in self.processes:
                p.join()

            self._flush_queue()  # clear queue contents

    def __iter__(self):
        for _ in range(self.count):
            yield self.q.get()
        self._join()


SEQ_LENGTH = 110
SEQ_DIM = 30
NUM_LABEL = 7

class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, init_states, threads):
        super(OCRIter, self).__init__()
        self.gen = GenPlate("plate/font/platech.ttf", 'plate/font/platechar.ttf', "plate/NoPlates", "plate/images/template.bmp")
        self.batch_size = batch_size
        self.init_states = init_states
        self.init_state_names = [x[0] for x in self.init_states]
        self.init_state_arrays = [np.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, SEQ_LENGTH * SEQ_DIM))] + init_states
        self.provide_label = [('label', (self.batch_size, NUM_LABEL))]

        def gen_batch_helper():
            return self._gen_batch()

        self.prefetch = Prefetch(count, threads, gen_batch_helper)

    def _gen_batch(self):
        data = []
        label = []
        for i in range(self.batch_size):
            plate_str = self.gen.genPlateString(-1, -1)
            img = self.gen.generate(plate_str)
            img = cv2.resize(img, (SEQ_LENGTH, SEQ_DIM))
            img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
            img = img.transpose(1, 0)
            img = img.reshape((SEQ_LENGTH * SEQ_DIM))
            img = np.multiply(img, 1/255.0)
            data.append(img)
            label.append(CTC.label_encode(plate_str))
        data_all = [np.array(data)] + self.init_state_arrays
        label_all = [np.array(label)]
        data_names = ['data'] + self.init_state_names
        label_names = ['label']
        data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
        return data_batch

    def reset(self):
        self.prefetch._reset()

    def join(self):
        self.prefetch._join()

    def __iter__(self):
        for batch in self.prefetch:
            batch.data = [mx.nd.array(d) for d in batch.data]
            batch.label = [mx.nd.array(d) for d in batch.label]
            yield batch

def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret

class CTC:
    @staticmethod
    def label_encode(label):
        ret = np.zeros(NUM_LABEL)
        ret[0] = index[label[:3]] + 1
        for i in range(NUM_LABEL - 1):
            ret[i + 1] = index[label[i + 3]] + 1
        return ret

    @staticmethod
    def label_decode(enc):
        enc = remove_blank(enc)
        pred = ''
        for c in enc:
            pred += str((chars[int(c - 1)]))
        return pred

    @staticmethod
    def ctc_label(p):
        ret = []
        p1 = [0] + p
        for i in range(len(p)):
            c1 = p1[i]
            c2 = p1[i+1]
            if c2 == 0 or c2 == c1:
                continue
            ret.append(c2)
        return ret

    @classmethod
    def ctc_decode(cls, pred, seq_len):
        p = []
        for k in range(seq_len):
            p.append(np.argmax(pred[k]))

        enc = cls.ctc_label(p)
        pred = cls.label_decode(enc)

        return pred

def Accuracy(label, pred):
    batch_size_per_gpu = pred.shape[0] / SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(batch_size_per_gpu):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * batch_size_per_gpu + i]))
        p = CTC.ctc_label(p)
        hit += ed.eval(l, p)  # edit distance
        total += len(l)
    return 1 - hit / total

if __name__ == '__main__':
    batch_size = 1024
    num_hidden = 128

    num_epoch = 100
    learning_rate = 0.001
    momentum = 0.9

    prefetch_thread = 16

    train_from_scratch = False
    prefix = 'model/ocr-plate'
    epoch = 85

    contexts = [mx.context.gpu(i) for i in range(4)]

    init_c = [('f_init_c', (batch_size, num_hidden)), ('b_init_c', (batch_size, num_hidden))]
    init_h = [('f_init_h', (batch_size, num_hidden)), ('b_init_h', (batch_size, num_hidden))]
    init_states = init_c + init_h

    data_train = OCRIter(1000, batch_size, init_states, prefetch_thread)
    data_val = OCRIter(50, batch_size, init_states, prefetch_thread)

    symbol = blstm_unroll(SEQ_LENGTH, num_hidden, NUM_LABEL, len(chars) + 1)

    if train_from_scratch:
        model = mx.model.FeedForward(ctx=contexts,
                                     symbol=symbol,
                                     num_epoch=num_epoch,
                                     learning_rate=learning_rate,
                                     momentum=momentum,
                                     wd=0.00001,
                                     initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
    else:
        model = mx.model.FeedForward.load(prefix=prefix,
                                          epoch=epoch,
                                          ctx=contexts,
                                          num_epoch=num_epoch,
                                          learning_rate=learning_rate,
                                          momentum=momentum,
                                          wd=0.00001)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Accuracy),
              batch_end_callback=mx.callback.Speedometer(batch_size, 10),
              epoch_end_callback=mx.callback.do_checkpoint(prefix))

    data_train.join()
    data_val.join()
