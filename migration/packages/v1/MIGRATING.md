* train.py
    * MonitoredTrainingSession:
        *  https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/MonitoredTrainingSession


* LSTM:
    * WARNING:tensorflow:From /workspaces/GeoTrackNet/migration/packages/v1/src/migration/models/vrnn.py:345: LSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.

 * Initializers:
    * tf.compat.v1.keras.initializers.VarianceScaling is also in keras for use in layers https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
    * use https://www.tensorflow.org/api_docs/python/tf/zeros_initializer
Instructions for updating:
This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
