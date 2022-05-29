# delete models that doesn't satisify the criteria before reaching the maximum trials
from os import walk
import core.tools as tools
from shutil import rmtree

class Model_seletor():
    def delete(self, model_dir, sub_dir='/', max_trials=1e6):
        '''
        say there's a file structure:
        - model_dir
          - model_0
            - dir_0
            - dir_1
            - hp.json
            - log.json
            - model.pth
          - model_1
        This function will delete all model_i which has log.json indicate the model is trained more than the maximum trials
        '''
        root, dirs, _ = next(walk(model_dir))
        for dir_name in dirs:
            log = tools.load_log(root + '/' + dir_name + sub_dir)
            if log['trials'][-1] >= max_trials:
                rmtree(root + '/' + dir_name)
