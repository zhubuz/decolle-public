import numpy as np
import torch
import importlib
import decolle.spikeIO as io
from torch.utils.data import Dataset, DataLoader

def augmentData(event):
    xs = 8
    ys = 8
    th = 10
    xjitter = np.random.randint(2 * xs) - xs
    yjitter = np.random.randint(2 * ys) - ys
    ajitter = (np.random.rand() - 0.5) * th / 180 * 3.141592654
    sinTh = np.sin(ajitter)
    cosTh = np.cos(ajitter)
    event[:, 0] = event[:, 0] * cosTh - event[:, 1] * sinTh + xjitter
    event[:, 1] = event[:, 0] * sinTh + event[:, 1] * cosTh + yjitter
    return event

class IBMGestureDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength, augment=False):
        self.path = datasetPath
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)
        self.augment = augment

        with open(sampleFile) as f:
            # self.samples = f.readlines()
            samples = f.read().splitlines()

        # only use even classes {0, 2, 4, 6, 8, 10} -> {0, 1, 2, 3, 4, 5}
        self.samples = []
        self.labels = []
        for filename in samples:
            fullLabel = int(filename.split('/')[-1].split('.')[0])
            if fullLabel % 2 == 0:
                self.samples.append(filename)
                self.labels.append(fullLabel // 2)

    def __getitem__(self, index):
        # Read inoput and label
        filename = self.samples[index]
        classLabel = self.labels[index]

        # print(filename, classLabel)

        npEvent = np.load(self.path + filename)

        # Reduce the spatial dimension by 4 and remove polarity
        npEvent[:, 0] = npEvent[:, 0] // 4
        npEvent[:, 1] = npEvent[:, 1] // 4
        #npEvent[:, 2] = 0

        if self.augment is True:
            npEvent = augmentData(npEvent)
        # Read input spike
        inputSpikes = io.event(
            npEvent[:, 0], npEvent[:, 1], npEvent[:, 2], npEvent[:, 3]
        ).toSpikeTensor(torch.zeros((2, 32, 32, self.nTimeBins)),
                        samplingTime=self.samplingTime,
                        randomShift=self.augment)
        # Create one-hot encoded desired matrix
        desiredClass = torch.zeros((self.nTimeBins, 6))
        desiredClass[:, classLabel] = 1

        inputSpikes = inputSpikes.permute(3, 0, 1, 2)
        #print('inputspikes', inputSpikes.shape)
        #return inputSpikes, desiredClass, classLabel
        return inputSpikes, desiredClass

    def __len__(self):
        return len(self.samples)