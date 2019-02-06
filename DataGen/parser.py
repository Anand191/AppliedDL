import os
import json
import numpy as np

def create_filepaths(master_path):
    folders = os.listdir(master_path)
    files = []
    for fd in folders:
        all_files = os.listdir(os.path.join(master_path, fd))
        all_paths = [os.path.join(master_path, fd, fl) for fl in all_files]
        files.extend(all_paths)
    return files

def create_dialogs(files):
    dialogues = []
    for tf in files:
        with open(tf) as f:
            var = json.load(f)
            uu, sr, label = [], [], []
            for turn in var["turns"]:
                if turn["speaker"] == "U":
                    uu.append(turn["utterance"])
                else:
                    sr.append(turn["utterance"])
                    majority_label = {'NB': 0, 'PB': 0, 'B': 0}
                    for annotate in turn["annotations"]:
                        if annotate["breakdown"] == "O":
                            majority_label['NB'] += 1
                        elif annotate["breakdown"] == "T":
                            majority_label['PB'] += 1
                        else:
                            majority_label['B'] += 1
                    label.append(max(majority_label, key=lambda key: majority_label[key]))
            if (len(sr) > len(uu)):
                for i in range(len(sr) - len(uu)):
                    sr.pop(i)
                    label.pop(i)
            elif (len(uu) > len(sr)):
                for i in range(len(sr), len(uu)):
                    uu.pop(i)
            dialogue_until = ['']
            uu_sr = [uu[i] + ' ' + s for i, s in enumerate(sr)]
            dialog = np.c_[uu, sr, uu_sr, label]
        dialogues.append(dialog)
        # for d in dialog:
        #     print(*d, sep="-------")

    return dialogues


master_train = '../resources/DBDC3/DBDC3/dbdc3/en/dev'
master_eval = '../resources/DBDC3/DBDC3/dbdc3/en/eval'

train_files = create_filepaths(master_train)
eval_files = create_filepaths(master_eval)

train_dialogues = create_dialogs(train_files)
eval_dialogues = create_dialogs(eval_files)

assert(len(train_dialogues) == len(train_files))
assert(len(eval_dialogues) == len(eval_files))






