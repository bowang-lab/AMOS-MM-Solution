import os
import pandas as pd
from tqdm import tqdm
# from GREEN.green_score import GREEN
from GREEN.green_score import GREEN
import json
import re

class GenerateGreenScore:
    def __init__(self, csv_path, cache_dir=None, save_every=10, organs=["abdomen", "chest", "pelvis"]): 
        self.csv_path = csv_path
        self.save_every = save_every
        self.organs = organs
        self.model = GREEN(
            model_name="StanfordAIMI/GREEN-radllama2-7b",
            output_dir=cache_dir,
            cache_dir=cache_dir
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
        for organ in self.organs:
            mask = (
                (self.df[f"green_{organ}"] == -1)
                & (self.df[f"gt-{organ}"].notna())
                & (self.df[f"generated-{organ}"].notna())
                & (self.df[f"gt-{organ}"].astype(bool))
                & (self.df[f"generated-{organ}"].astype(bool))
            )

            if not mask.any():
                continue

            sub_df = self.df.loc[mask, [f"gt-{organ}", f"generated-{organ}"]]

            refs = sub_df[f"gt-{organ}"].tolist()
            hyps = sub_df[f"generated-{organ}"].tolist()

            mean, std, green_score_list, summary, result_df = self.model(
                refs=refs,
                hyps=hyps
            )

            self.df.loc[sub_df.index, f"green_{organ}"] = green_score_list
            self.df.loc[sub_df.index, f"explanation_{organ}"] = result_df["green_analysis"]

            self.save_df()  
