# Warp CTC Example
Built from [mxnet warp ctc example](https://github.com/dmlc/mxnet/tree/master/example/warpctc).

## Added utilities:
- OCRIter prefetch to boost training speed
- Pre-trained model, accuracy: 97.6%
- OCR model prediction

## Demo:
Mxnet==0.8.0 with baidu warpctc plugin should be installed first:
```
python infer_ocr.py
```
