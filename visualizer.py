# Visualizer #

import matplotlib.pyplot as plt
import librosa
import numpy

class visualizer(object):
    
    def __init__(self):
        # create pyplot
        self.plt = plt
        # set figure size , 
        self.fig = self.plt.figure(figsize=(30, 10))
        # width & height space
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)
   
    # Signal Plot

    def signal_plot(self,y):
        ax = self.fig.add_subplot(1,1,1)
        ax.plot(y);
        ax.set_title('Signal');
        ax.set_xlabel('Time (samples)');
        ax.set_ylabel('Amplitude');

    # Spectrum Plot

    def spec_plot(self,ft):
        ax = self.fig.add_subplot(1,1,1)
        ax.plot(ft)
        ax.set_title('Spectrum');
        ax.set_xlabel('Frequency Bin');
        ax.set_ylabel('Amplitude');
    
    # Spectrum Power Plot

    def spec_power_plot(self,S,sr):
        log_S = librosa.power_to_db(S, ref=numpy.max)
        ax = self.fig.add_subplot(1,1,1)
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        ax.title('mel power spectrogram')
        ax.colorbar(format='%+02.0f dB')
        ax.tight_layout()

    # Loss Plot

    def loss_plot(self, loss, val_loss):
        # create subpolt
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Test"], loc="upper right")


    # Save Figure
    
    def save_figure(self, name):
        # save fiure
        self.plt.savefig(name)