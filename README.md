# Warp CTC Example
Built from [mxnet warp ctc example](https://github.com/dmlc/mxnet/tree/master/example/warpctc)

Car plate generation code imported from [end-to-end-for-chinese-plate-recognition](https://github.com/szad670401/end-to-end-for-chinese-plate-recognition)

## Added utilities:
- OCRIter prefetch to boost training speed
- OCR model prediction
- Blstm implementation

## Pre-trained model:
- Captcha model, accuracy 97.6%
- Chinese car plate model, accuracy 98.2% (computed by edit distance)

## Demo:
Mxnet==0.9.5 with baidu warpctc plugin should be installed first:
```
python infer_ocr_captcha.py
python infer_ocr_plate.py
```
