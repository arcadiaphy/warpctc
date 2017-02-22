import numpy as np
import mxnet as mx
from PIL import Image

from ocr_plate import CTC, OCRIter

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
    num_hidden = 128
    num_label = 7
    seq_len = 110
    img_dim = seq_len * 30

    init_c = [('f_init_c', (batch_size, num_hidden)), ('b_init_c', (batch_size, num_hidden))]
    init_h = [('f_init_h', (batch_size, num_hidden)), ('b_init_h', (batch_size, num_hidden))]
    init_states = init_c + init_h
    inputs = [('data', (batch_size, img_dim)), ('label', (batch_size, num_label))]
    input_shapes = dict(init_states + inputs)

    data = OCRIter(batch_num, batch_size, init_states, 1)

    predictor = Predictor('model/ocr-plate', 85, seq_len, input_shapes)

    for batch in data:
        prob = predictor.forward(batch.data[0])
        print 'label:\t\t' + CTC.label_decode(batch.label[0].asnumpy()[0])
        print 'predicted:\t' + CTC.ctc_decode(prob, seq_len)

        img = batch.data[0].asnumpy().reshape((seq_len, 30)).transpose((1, 0))
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.show()
