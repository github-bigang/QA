"""
Module to score BiMPM prediction
"""
import sys
import operator

def print_top_errors(confusion_matrix, top_relation_error, top_n_error = 10):
	confusion_matrix_sorted = sorted(confusion_matrix.items(), key=operator.itemgetter(1), reverse= True)
	print '=== gold -- prediction -- count ==='
	top_n = top_n_error
	for ((gold, prediction), count) in confusion_matrix_sorted:
		print '%s -- %s -- %s' % (gold, prediction, str(count))
		top_n -= 1
		if top_n < 0:
			break
	top_error_sorted = sorted(top_relation_error.items(), key=operator.itemgetter(1), reverse= True)		
	print '====== top relation errors ======'
	top_n = top_n_error
	for (gold, count) in top_error_sorted:
		print gold, str(count)
		top_n -= 1
		if top_n < 0:
			break

def add_one_to_map_entry(map, key):
	try:
		map[key] += 1
	except KeyError:
		map[key] = 1
			
def record_error(confusion_matrix, top_relation_error, gold_tags, prediction):
	for gold in gold_tags:
		add_one_to_map_entry(confusion_matrix, (gold, prediction))
		add_one_to_map_entry(top_relation_error, gold)

def get_confidence(prob_string):
	tokens = prob_string.split()
	for t in tokens:
		if t.startswith('1:'):
			return float(t[t.find(':') + 1 :])
	return 0.0
	
prediction_file = sys.argv[1]
prediction_lines = [line.rstrip('\n') for line in open(prediction_file)]

prediction = dict()

for line in prediction_lines:
	tokens = line.split('\t')
	qid_head = tokens[5]
	question = tokens[3]
	gold_tag = tokens[4] if tokens[0] == '1' else None
	predicted_tag = tokens[4]
	prediction_confidence = get_confidence(tokens[2])
	
	try:
		(q, gt, pt, pt_conf) = prediction[qid_head]
		if not gt:
			gt = [gold_tag] if gold_tag else None
		elif gt and gold_tag:
			gt.append(gold_tag)
		if prediction_confidence > pt_conf:
			pt = predicted_tag
			pt_conf = prediction_confidence
		prediction[qid_head] = (q, gt, pt, pt_conf)
	except KeyError:
		gt = [gold_tag] if gold_tag else None
		prediction[qid_head] = (question, gt, predicted_tag, prediction_confidence) 
	
total = 0
correct = 0

confusion_matrix = dict()
top_relation_error = dict()

for key, _ in prediction.iteritems():
	(q, gt, pt, pt_conf) = prediction[key]
	total = total + 1
	if gt and pt in gt:
		correct = correct + 1  
	else:
		print key, (q, gt if gt is not None else [], pt, pt_conf)
# 		record_error(confusion_matrix, top_relation_error, gt, pt)

print_top_errors(confusion_matrix, top_relation_error, 20)		
print 'total -', total, 'correct -', correct, '% -', (100.0 * correct / total)