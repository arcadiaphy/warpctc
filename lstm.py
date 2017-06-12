import mxnet as mx
from collections import namedtuple

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx):
    """LSTM Cell symbol"""
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def lstm_unroll(num_lstm_layer, seq_len, num_hidden, num_label, num_class):
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_class)

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data = label, dtype = 'int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_length = num_label, input_length = seq_len)
    return sm

# unlike lstm_unroll, remove layered lstm structure
def blstm_unroll(seq_len, num_hidden, num_label, num_class):
    params = LSTMParam(i2h_weight=mx.sym.Variable("i2h_weight"),
                       i2h_bias=mx.sym.Variable("i2h_bias"),
                       h2h_weight=mx.sym.Variable("h2h_weight"),
                       h2h_bias=mx.sym.Variable("h2h_bias"))
    forward_states = LSTMState(c=mx.sym.Variable("f_init_c"),
                               h=mx.sym.Variable("f_init_h"))
    backward_states = LSTMState(c=mx.sym.Variable("b_init_c"),
                                h=mx.sym.Variable("b_init_h"))

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len)

    # forward, layeridx 0
    forward_hidden_all = []
    for seqidx in range(seq_len):
        next_states = lstm(num_hidden, indata=wordvec[seqidx],
                           prev_state=forward_states,
                           param=params,
                           seqidx=seqidx, layeridx=0)
        forward_hidden_all.append(next_states.h)
        forward_states = next_states

    # backward, layeridx 1
    backward_hidden_all = []
    for seqidx in reversed(range(seq_len)):
        next_states = lstm(num_hidden, indata=wordvec[seqidx],
                           prev_state=backward_states,
                           param=params,
                           seqidx=seqidx, layeridx=1)
        backward_hidden_all.append(next_states.h)
        backward_states = next_states
    backward_hidden_all.reverse()

    hidden_all = [mx.sym.Concat(*p) for p in zip(forward_hidden_all, backward_hidden_all)]
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_class)

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data = label, dtype = 'int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_length = num_label, input_length = seq_len)
    return sm
