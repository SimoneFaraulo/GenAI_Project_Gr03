import time
from math import isnan

class ProgressIndicator:
    def __init__(self, dataset_length, initial_epoch=0,
                 message_period=30.0):
        self.dataset_length=dataset_length
        self.epoch_number=initial_epoch
        self.message_period=message_period
        self.last_message_time=0.0
        self.count=0
        self.lcount=0
        self.sums={ }
        self.lsums={ }

    def start_new_epoch(self, epoch_number=None):
        if epoch_number is None:
            self.epoch_number += 1
        else:
            self.epoch_number = epoch_number
        self.average_loss=None
        self.last_message_time=0.0
        self.count=0
        self.sums={ }

    def update(self, batch_size, batch_loss=None, **kwargs):
        if batch_loss is not None:
            kwargs['batch_loss']=batch_loss
        now=time.time()
        dt=now-self.last_message_time
        self.count += batch_size
        self.lcount += batch_size
        for k in kwargs:
            prev=self.sums.get(k, 0)
            self.sums[k] = prev + batch_size*float(kwargs[k])
            lprev=self.lsums.get(k, 0)
            self.lsums[k] = lprev + batch_size*float(kwargs[k])
        if self.count>=self.dataset_length or dt>=self.message_period:
            self.show_message()
            self.last_message_time=now

    def show_message(self):
        ratio=self.count*100.0/self.dataset_length
        msg=[f'Epoch={self.epoch_number:3d} |{ratio:8.4f}%|']
        if self.count>0:
            for k in self.sums:
                val=self.sums[k]/self.count
                msg.append(f'{k}={val:7g}')
        msg=' '.join(msg)
        print(msg)
        if self.lcount>0:
            msg=['.'*13+'current:']
            for k in self.lsums:
                val=self.lsums[k]/self.lcount
                msg.append(f'{k}={val:7g}')
            self.lcount=0
            self.lsums={ }
            msg=' '.join(msg)
            print(msg)
