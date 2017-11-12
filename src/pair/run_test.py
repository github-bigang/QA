from Decoder import Decoder
import sys

test = Decoder('data/small/answer.txt', 'data/small/answer-seg.txt', 'data/small/question.txt', 'data/small/question-seg.txt', 'data/small/question_answer.txt')
results = test.answer(sys.argv[1], 3)
for score, answer, qid in results:
    print(str(score) + '\t' + answer + '\t' + qid)

