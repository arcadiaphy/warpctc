import cv2
import Queue
import random
import numpy as np
import mxnet as mx
from multiprocessing import Process, Manager, Value
from captcha.image import ImageCaptcha

from lstm import lstm_unroll

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

def gen_rand():
    buf = ""
    max_len = random.randint(3, 4)
    for i in range(max_len):
        buf += str(random.randint(0, 9))
    return buf

class Prefetch:
    def __init__(self, count, worker, f):
        self.m = Manager()
        self.q = self.m.Queue(worker)
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
        self.stop.value = True
        self._flush_queue()  # ensure all processes not blocked

        for p in self.processes:
            p.join()

        self._flush_queue()  # clear queue contents

    def __iter__(self):
        for _ in range(self.count):
            yield self.q.get()
        self._join()

class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, init_states, threads):
        super(OCRIter, self).__init__()
        self.captcha = ImageCaptcha(fonts=['plate/font/platechar.ttf'])
        self.batch_size = batch_size
        self.init_states = init_states
        self.init_state_names = [x[0] for x in self.init_states]
        self.init_state_arrays = [np.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, 2400))] + init_states
        self.provide_label = [('label', (self.batch_size, 4))]

        def gen_batch_helper():
            return self._gen_batch()

        self.prefetch = Prefetch(count, threads, gen_batch_helper)

    def _gen_batch(self):
        data = []
        label = []
        for i in range(self.batch_size):
            num = gen_rand()
            img = self.captcha.generate(num)
            img = np.fromstring(img.getvalue(), dtype='uint8')
            img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (80, 30))
            img = img.transpose(1, 0)
            img = img.reshape((80 * 30))
            img = np.multiply(img, 1/255.0)
            data.append(img)
            label.append(CTC.label_encode(num))
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

SEQ_LENGTH = 80

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
        ret = np.zeros(4)
        for i in range(len(label)):
            ret[i] = 1 + int(label[i])
        if len(label) == 3:
            ret[3] = 0
        return ret

    @staticmethod
    def label_decode(enc):
        enc = remove_blank(enc)
        pred = ''
        for c in enc:
            pred += str((int(c) - 1))
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
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total

if __name__ == '__main__':
    batch_size = 128
    num_hidden = 100
    num_lstm_layer = 2

    num_epoch = 10
    learning_rate = 0.001
    momentum = 0.9
    num_label = 4

    prefetch_thread = 16

    train_from_scratch = True
    prefix = 'model/ocr-captcha'
    epoch = 15

    contexts = [mx.context.gpu(i) for i in range(4)]

    init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = OCRIter(10000, batch_size, init_states, prefetch_thread)
    data_val = OCRIter(1000, batch_size, init_states, prefetch_thread)

    if train_from_scratch:
        symbol = lstm_unroll(num_lstm_layer, SEQ_LENGTH, num_hidden, num_label, 11)
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
              eval_metric=mx.metric.np(Accuracy),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),
              epoch_end_callback=mx.callback.do_checkpoint(prefix))

    data_train.join()
    data_val.join()
