
import os
import json
import time
import math
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
from core.live_data_processor import LiveDataLoader
from core.model import Model


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #  Setting pyplot fig size
    plt.rcParams['figure.figsize'] = [10, 8]
    # Download gold price from yahoo finance
    gld_dataframe = yf.download(tickers="GLD", start="2000-1-1")

    data = LiveDataLoader(
        dataframe=gld_dataframe,
        split=0.8,
        cols=['Open', 'Close', 'Volume']
    )

    model = Model()
    model.build_model()
    model.model.summary()

    '''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	'''
    # out-of memory generative training
    steps_per_epoch = math.ceil(((data.len_train - 30) / 32))

    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=30,
            batch_size=32,
            normalise=True
        ),
        epochs=2,
        batch_size=32,
        steps_per_epoch=steps_per_epoch,
        save_dir='saved_models'
    )



    x_test, y_test = data.get_test_data(
        seq_len=30,
        normalise=True)

    predictions_p = model.predict_point_by_point(x_test)
    plot_results(predictions_p, y_test)
    #
    # # predicted data that are below 3 difference
    # diff = abs(predictions_p[:,0] - y_test) < 3
    #
    # count = 0
    # for element in diff:
    #     if element == True:
    #         count = count + 1
    #
    # # percentage of <5000 difference data in the whole testing data set
    # percentage = count/len(diff)
