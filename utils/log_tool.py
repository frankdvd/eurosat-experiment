import logging
import os



class Logger():

    def __init__(self, taskname, mec):
        if not os.path.exists('./log'):
            os.mkdir('./log')

        logging.basicConfig(level=logging.INFO, filename='./log/'+taskname, filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(taskname+" Logger")
        self.logger.info("="*20 + " Model MEC = %d " %(mec) + "="*20)

        self.mec = mec
        self.best_train_acc = 0
        self.best_vali_acc = 0
        self.best_test_acc = 0

    def log(self, epoch, acc, run_type):
        self.logger.info("MEC: %d | epoch %d | %s Acc: %.3f" %(self.mec ,epoch, run_type, acc))
        if run_type == 'train':
            self.best_train_acc = max(self.best_train_acc, acc)
        elif run_type == 'test':
            self.best_test_acc = max(self.best_test_acc, acc)
        else:
            self.best_vali_acc = max(self.best_vali_acc, acc)

    def finalize_log(self):
        self.logger.info("MEC: %d | Best Train Acc: %.3f | Best Validation Acc: %.3f | Best Test Acc: %.3f" %(self.mec, self.best_train_acc, self.best_vali_acc, self.best_test_acc))


def read_log(filepath):
    result = {}
    best_result = {}
    current_mec = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.split()[-1][0] == '=':
                result[int(line.split()[-2])] = {'train':[], 'validation':[], 'test':[]}
                best_result[int(line.split()[-2])] = {'train':0, 'validation':0, 'test':0}
                current_mec = int(line.split()[-2])
            else:
                if len(line.split())<=18:
                    result[current_mec][line.split()[-3]].append(float(line.split()[-1]))
                else:
                    best_result[current_mec]['train'] = float(line.split()[-11])
                    best_result[current_mec]['validation'] = float(line.split()[-6])
                    best_result[current_mec]['test'] = float(line.split()[-1])

    return result, best_result
    



