
Adaptation of the PredRNN using TensorFlow Keras.

The PredRNN was designed for precipitation nowcasting.

This is an adaptation/generalization to wind speed forecasting.

## brief explanation

# get_reanalysis.sh

download relevant reanalysis data

# rean2txt.sh

prepare data in a suitable format (uses cdo)

# txt2tensor.py

convert the data to a numpy compatible tensor

# predrnn_training.py

neural network definition and training and saving parameters

# predrnn_forecast.py

loading parameters and making forecasts + print statistics

# draw_maps.py

loading parameters and making forecasts + draw maps

