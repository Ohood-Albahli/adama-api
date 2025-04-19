from Train_model import AdamaTrainer
from Adama_model import csv_path, data_dir

trainer = AdamaTrainer(csv_path, data_dir)
trainer.train()

 
from Test_model import AdamaTester
from Adama_model import csv_path, data_dir

tester = AdamaTester(csv_path, data_dir)
tester.test()
