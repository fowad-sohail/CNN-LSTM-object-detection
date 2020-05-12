# Object Detection in Videos Using a CNN/LSTM Architecture

A CNN/LSTM model trained on the TinyTLP V2 dataset for object detection in video data.

You can find the dataset [here](https://gist.github.com/amoudgl/2193261e6b6f7e2a3aeace42b3894b5b).

[objectdetectionV1.py](https://github.com/fowad-sohail/CNN-LSTM-object-detection/blob/master/objectdetectionV1.py) is the first iteration of this model, which only trains on two videos and tests on one. Because of this small dataset, it performs poorly (~40% accuracy). This was written originally using Google Colab and later training on Rowan AI's lambda01 machine.

[finalModel.py](https://github.com/fowad-sohail/CNN-LSTM-object-detection/blob/master/finalModel.py) is the second iteration of this model. It trains on 33 videos and tests on 14. It utilizes a generator, found in [sequencer.py](https://github.com/fowad-sohail/CNN-LSTM-object-detection/blob/master/sequencer.py) to load the data in batches, instead of loading it all in upfront. It was designed in this way because of the large dataset and to preserve memory resources.
