
The [PredRNN](https://github.com/kami93/PredRNN) was [designed](https://dl.acm.org/doi/10.5555/3294771.3294855) for precipitation nowcasting.

This is an adaptation/generalization for wind speed forecasting.

This repo accompanies our submitted paper entitled "On data selection for wind forecasting neural networks training".

## get_reanalysis.sh

download relevant reanalysis data

## rean2txt.sh

prepare data in a suitable format (uses cdo)

## txt2tensor.py

convert the data to a numpy compatible tensor

## predrnn_training.py

neural network definition and training and saving parameters

## predrnn_forecast.py

loading parameters and making forecasts + print statistics

## draw_maps.py

loading parameters and making forecasts + draw maps

