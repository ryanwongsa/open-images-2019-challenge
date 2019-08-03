
import neptune
from train import save_components

class Callback(object):
    def __init__(self, project_name, experiment_name, hyper_params, save_dir):
        neptune.init(project_name)
        self.experiment = neptune.create_experiment(name=experiment_name,
            params=hyper_params)
        self.save_dir = save_dir

    def on_train_start(self, data_dict):
        print("Started Training")

    def on_epoch_start(self, data_dict):
        print("Started Epoch:", data_dict["epoch"])
    
    def on_batch_end(self, data_dict):
        if (data_dict["batch_num"]%10 == 0):
            self.experiment.send_metric('batch_loss', data_dict["batch_num"], data_dict["loss"])

    def on_end_epoch(self, data_dict):
        self.experiment.send_metric('train_epoch_loss', data_dict["epoch_num"], data_dict["epoch_loss"])
        self.experiment.send_metric('valid_epoch_loss', data_dict["epoch_num"], data_dict["eval_loss"])
        img_pils = data_dict["display_imgs"]
        for img_pil in img_pils:
            self.experiment.send_image("eval_images", img_pil)


    def on_train_end(self, data_dict):
        self.experiment.log_metric("mAp", data_dict["mAp"])
        trainer = data_dict["trainer"]
        save_components(trainer.model, trainer.optimizer, trainer.scheduler, self.save_dir)
        self.experiment.stop()