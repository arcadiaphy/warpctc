import numpy as np
import mxnet as mx
from PIL import Image

from ocr_captcha import CTC, OCRIter

class Predictor:
    def __init__(self, prefix, epoch, seq_len, input_shapes):
        self.seq_len = seq_len
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        self.sym = sym
        self.exc = self.sym.simple_bind(ctx=mx.cpu(0), grad_req='null', **input_shapes)

        for k in arg_params.keys():
            arg_params[k].copyto(self.exc.arg_dict[k])

        for k in aux_params.keys():
            aux_params[k].copyto(self.exc.aux_dict[k])

    def forward(self, img):
        if isinstance(img, np.ndarray):
            img = mx.nd.array(img)
        img.copyto(self.exc.arg_dict['data'])
        return self.exc.forward()[0].asnumpy()


if __name__ == '__main__':
    batch_num = 1
    batch_size = 1
    num_hidden = 100
    num_lstm_layer = 2
    num_label = 4
    seq_len = 80
    img_dim = 80 * 30

    init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    inputs = [('data', (batch_size, img_dim)), ('label', (batch_size, num_label))]
    input_shapes = dict(init_states + inputs)

    data = OCRIter(batch_num, batch_size, init_states, 1)

    predictor = Predictor('model/ocr-captcha', 15, seq_len, input_shapes)

    for batch in data:
        prob = predictor.forward(batch.data[0])
        print 'label: ' + CTC.label_decode(batch.label[0].asnumpy()[0])
        print 'predicted: ' + CTC.ctc_decode(prob, seq_len)

        img = batch.data[0].asnumpy().reshape((80, 30)).transpose((1, 0))
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.show()
