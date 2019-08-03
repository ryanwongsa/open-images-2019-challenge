
import neptune
from train import save_components
import time
import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import PIL

class Callback(object):
    def __init__(self, project_name, experiment_name, hyper_params, save_dir):
        neptune.init(project_name)
        self.experiment = neptune.create_experiment(name=experiment_name,
            params=hyper_params, upload_source_files=[])
        self.save_dir = save_dir

    def on_train_start(self, data_dict):
        self.start_train = time.time()
        print("Started Training")
        
    def on_epoch_start(self, data_dict):
        print("Started Epoch:", data_dict["epoch"])
        self.start = time.time()

    def on_end_train_epoch(self, data_dict):
        print("Ended Train Epoch in (hours):", time.time() - self.start)/3600.0)
    
    def on_batch_end(self, data_dict):
        if (data_dict["batch_num"]%10 == 0):
            self.experiment.send_metric('batch_loss', data_dict["batch_num"], data_dict["loss"])
            print(datetime.datetime.now(), ":", data_dict["batch_idx"],"/", data_dict["num_batches"])

    
    def on_end_epoch(self, data_dict):
        self.experiment.send_metric('train_epoch_loss', data_dict["epoch_num"], data_dict["epoch_loss"])
        self.experiment.send_metric('valid_epoch_loss', data_dict["epoch_num"], data_dict["eval_loss"])
        img_pils = data_dict["display_imgs"]
        for img_pil in img_pils:
            self.experiment.send_image("eval_images", img_pil)
        self.end = time.time()
        self.experiment.log_metric("mAp", data_dict["mAp"])
        print("Epoch completed in (hours):",(self.end - self.start)/3600.0)
        img_pil = aps_img(data_dict["dict_aps"])
        self.experiment.send_image("MAPs", img_pil)


    def on_train_end(self, data_dict):
        self.experiment.log_metric("final_mAp", data_dict["mAp"])
        trainer = data_dict["trainer"]
        save_components(trainer.model, trainer.optimizer, trainer.scheduler, self.save_dir)
        self.end_train = time.time()
        print("Training completed in (hours):",(self.end_train - self.start_train)/3600.0)
        self.experiment.send_artifact('temp/final.pth')
        self.experiment.stop()


def aps_img(dict_aps):
    xticks = []
    values = []
    for key, value in dict_aps.items():
        values.append(value)
        xticks.append(key)
        
    values = np.array(values)
    xticks = np.array(xticks)

    index = values.argsort()
    values = values[index][::-1]
    xticks = xticks[index][::-1]

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(20,80))
    canvas = FigureCanvas(fig)
    ax.barh(range(len(values)), values)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(xticks)
    ax.invert_yaxis() 
    ax.set_xlabel('AP')
    ax.set_title('mAp')
    plt.tight_layout()
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    plt.close()
    return pil_image