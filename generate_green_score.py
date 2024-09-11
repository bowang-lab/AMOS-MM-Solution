import os
import pandas as pd
from tqdm import tqdm
from green_score import GREEN
import json

class GenerateGreenScore:
    def __init__(self, csv_path, cache_dir=None, save_every=10, organs=["chest"]): 
        # ["abdomen", "chest", "pelvis"]
        self.csv_path = csv_path
        self.save_every = save_every
        self.organs = organs
        self.model = GREEN(
            model_id_or_path="StanfordAIMI/GREEN-radllama2-7b",
            do_sample=False,  # should be always False
            batch_size=16,
            return_0_if_no_green_score=True,
            cuda=True,
            cache_dir=cache_dir,
            max_len=400
        )
        self.df = pd.read_csv(self.csv_path)
        self.df.fillna('', inplace=True)

        if "green_abdomen" not in self.df.columns.to_list():
            for organ in self.organs:
                self.df["green_" + organ] = [-1] * len(self.df)
                self.df["explanation_" + organ] = [""] * len(self.df)
        
    def run(self):
        self.generate_scores()
        self.save_df()
        self.save_summary()
        return self.df
        
    def save_df(self):
        self.df.to_csv(self.csv_path, index=False)
        
    def save_summary(self):
        summary = {}
        
        for organ in self.organs:
            o_ = [n for n in self.df[f'green_{organ}'].to_list() if n != -1]
            summary[f"{organ}"] = sum(o_) / len(o_)    
    
        path = f"{os.sep}".join(self.csv_path.split(os.sep)[:-1])
        with open(path + os.sep  + "summary.json", 'w') as json_file:
            json.dump(summary, json_file, indent=4)
        
    def generate_scores(self):

        for indx in tqdm(self.df.index):
            row = self.df.iloc[indx]
            for organ in self.organs:
                if row[f"green_{organ}"] != -1:
                    continue
                if row[f"gt-{organ}"] and row[f"gt-{organ}"]:
                    _, green, explination = \
                        self.model(refs=[row[f"gt-{organ}"]], hyps=[row[f"generated-{organ}"]])
                    self.df[f"green_{organ}"].iloc[indx] = green[0].item()
                    self.df[f"explanation_{organ}"].iloc[indx] = explination[0]                

            if indx % self.save_every == 0:
                print(f"Saving at indx {indx}")
                self.save_df()  