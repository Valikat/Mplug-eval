import json

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
import argparse


if __name__ == "__main__":
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt_cap")
    parser.add_argument("-pred_cap")
    args = vars(parser.parse_args())

    with open('caption_result_custom_videos.json') as f:
        ground_truth = {}
        generated_caption = {}

        json = json.load(f)
        labelCaps = []
        outCap = []
        tempId = None
        for result in json:
            goldCap = result[0]["gold_caption"]
            fullGoldCap = ""
            for letter in result[0]["gold_caption"]:
                fullGoldCap = fullGoldCap + str(letter)
            id = result[0]["video_id"]
            if tempId == None:
                tempId = id
            if id != tempId:
                ground_truth[id] = labelCaps
                outCap.append(predCap)
                generated_caption[id] = outCap
                labelCaps = []
                outCap = []
            goldCap = fullGoldCap
            predCap = result[0]["pred_caption"]
            labelCaps.append(goldCap)
            # outCap = predCap
            tempId = id

    avg_bleu_score, bleu_scores = Bleu(4).compute_score(ground_truth, generated_caption)
    avg_cider_score, cider_scores = Cider().compute_score(ground_truth, generated_caption)
    avg_meteor_score, meteor_scores = Meteor().compute_score(ground_truth, generated_caption)
    avg_rouge_score, rouge_scores = Rouge().compute_score(ground_truth, generated_caption)

    print("BLEU:", avg_bleu_score)
    print("cIDER:", avg_cider_score)
    print("Meteor:", avg_meteor_score)
    print("rouge:", avg_rouge_score)
