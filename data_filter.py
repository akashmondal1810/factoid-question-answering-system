from csv import DictReader, DictWriter
from collections import defaultdict


kID = 'id'
kQUESTION = 'question'
kANSWER = 'correctAnswer'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'



if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("data/sci_train.csv", 'r')))
    cat_file = open("data/wiki_categories.txt", 'r')
    final_qs = []
    cats = []

    for line in cat_file:
        cats.append(line.strip())

    cats = set(cats)

    for x in train:
        if x['answer' + x['correctAnswer']] not in cats:
            continue

        sent = x['question'].lower()
        if len(sent) < 5 or 'tiebreaker' in sent:
            continue

        if sent.endswith("et al."):
            continue

        final_qs.append(x)

    print(len(train))
    print(len(final_qs))

    all_keys = ["id", "question", "correctAnswer", "answerA", "answerB", "answerC", "answerD"]
    o = DictWriter(open("data/filtered_train.csv", 'w'), all_keys, extrasaction="ignore")
    o.writeheader()
    for x in final_qs:
        o.writerow(x)



    
            
            
    
