from reader import reader_predict
import copy

data = [{"paragraphs": [{"qas": [{"question": "Were Scott Derrickson and Ed Wood of the same nationality?", \
    "is_impossible": False, \
        "answers": [{"text": "yes", "answer_start": -1}], \
        "id": "5a8b57f25542995d1e6f1371"}], \
            "context": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\" Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, actor, writer, producer, and director."}], \
                "title": "Doctor Strange (2016 film)"}]

new_data = copy.deepcopy(data)
question = "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"
context = "The Hork-Bajir Chronicles is the second companion book to the \"Animorphs\" series, written by K. A. Applegate. With respect to continuity within the series, it takes place before book #23, \"The Pretender\", although the events told in the story occur between the time of \"The Ellimist Chronicles\" and \"The Andalite Chronicles\". The book is introduced by Tobias, who flies to the valley of the free Hork-Bajir, where Jara Hamee tells him the story of how the Yeerks enslaved the Hork-Bajir, and how Aldrea, an Andalite, and her companion, Dak Hamee, a Hork-Bajir, tried to save their world from the invasion. Jara Hamee's story is narrated from the points of view of Aldrea, Dak Hamee, and Esplin 9466, alternating in similar fashion to the \"Megamorphs\" books. Animorphs is a science fantasy series of young adult books written by Katherine Applegate and her husband Michael Grant, writing together under the name K. A. Applegate, and published by Scholastic. It is told in first person, with all six main characters taking turns narrating the books through their own perspectives. Horror, war, dehumanization, sanity, morality, innocence, leadership, freedom and growing up are the core themes of the series."
new_data[0]['paragraphs'][0]['qas'][0]['question'] = question
new_data[0]['paragraphs'][0]['context'] = context

prediction = reader_predict(new_data)
answer = list(prediction.values())[0]
print(answer)
