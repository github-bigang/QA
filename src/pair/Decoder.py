from __future__ import print_function
from vocab_utils import Vocab
import namespace_utils

import tensorflow as tf
import SentenceMatchTrainer
from SentenceMatchModelGraph import SentenceMatchModelGraph
import codecs
import os
import sys
sys.path.append('src/ir/')

class Decoder:
    class segment:
        def __init__(self):
            LTP_DATA_DIR = 'resources/ltp_data_v3.4.0/'
            cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
            
            from pyltp import Segmentor
            self.segmentor = Segmentor()
            self.segmentor.load_with_lexicon(cws_model_path, '/path/to/your/lexicon')
            
        def seg(self, text):
            words = self.segmentor.segment(text)
            return words
    
        def destroy(self):
            self.segmentor.release()
            
        def segFile(self, infile, outfile):
            data = codecs.open(infile, 'r')
            out = codecs.open(outfile, 'w')#, 'utf-8'
            for line in data:
                fields = line.strip().split('\t')
                out.write(fields[0] + '\t' + '\t'.join([self.seg(fields[i]) for i in range(1, len(fields))]) + '\n')
            data.close()
            out.close()
    
    def __init__(self, aFile, aSegFile, qFile, qSegFile, qaFile):
        self.all_questions = self.readFile(qFile)
        self.all_questions_seg = self.readFile(qSegFile)
        self.all_questions_seg_set = {}
        for qid in self.all_questions_seg:
            self.all_questions_seg_set[qid] = set(self.all_questions_seg[qid].split(' '))
        self.all_answers = self.readFile(aFile)
        self.all_answers_seg = self.readFile(aSegFile)
        self.all_answers_seg_set = {}
        for aid in self.all_answers_seg:
            self.all_answers_seg_set[aid] = set(self.all_answers_seg[aid].split(' '))
        self.seg = self.segment()
        self.all_qas = self.readQAFile(qaFile)

    def readQAFile(self, f):
        d = {}
        with open(f, 'r') as data:
            for line in data:
                fields = line.strip().split('\t')
                if fields[0] not in d:
                    d[fields[0]] = []
                d[fields[0]].append(fields[1])
        return d
    
    def readFile(self, f):
        d = {}
        with open(f, 'r') as data:
            for line in data:
                fields = line.strip().split('\t')
                d[fields[0]] = fields[1]
        return d

    def answer_dl(self, qtext, K = 1):
        qseg = self.seg.seg(qtext)
        qsegset = set(qseg)
        tmpFile = '/tmp/q' + qtext
        with open(tmpFile, 'w') as out:
            for aid in self.all_answers_seg:
                if len(self.all_questions_seg_set[aid].intersection(qsegset)) > 1 and len(self.all_answers_seg_set[aid].intersection(qsegset)) > 1:
                    out.write('2' + '\t' + aid + '\t' + (' '.join(qseg)) + '\t' + self.all_answers_seg[aid] + '\n')
        self.decode('model/small/SentenceMatch.sample', tmpFile, '/tmp/out' + qtext, 'resources/w2v/w2v_cn_wiki_100.txt', 'prediction')
        topK = self.aggregateResults(qsegset, '/tmp/out' + qtext, K)
        return [(s, self.all_answers[i], i) for s,i in topK if i != '-1']
    
    def answer_ir(self, qtext, K = 1):
        qseg = self.seg.seg(qtext)
        qseg = ' '.join(qseg)
        from queryIndex_tfidf import QueryIndex
        q=QueryIndex('data/small/question-index.txt')
        docs = q.queryIndex(qseg, K)
        if docs:
            return [(q.sigmoid(d[0]), self.all_answers[self.all_qas[d[1]][0]], d[1]) for d in docs]
        else:
            return []
        
    def answer(self, qtext, K = 1):
        ir = self.answer_ir(qtext, K)
        dl = self.answer_dl(qtext, K)
        i,j=0,0
        ans = []
        while i < len(ir) or j < len(dl):
            if i >= len(ir): 
                ans.extend(dl[j:]) 
                break
            elif j >= len(dl):
                ans.extend(ir[i:])
                break
            elif ir[i][0] > dl[j][0]:
                ans.append(ir[i])
                i += 1
            else:
                ans.append(dl[j])
                j += 1
        return ans[:K]
    
    def aggregateResults(self, qsegset, outfile, K):
        topK = [(0, '-1')] * K
        with open(outfile, 'r') as data:
            for line in data:
                fields = line.strip().split('\t')
                score = float(fields[2].split(' ')[0][2:])
                intersect = len(qsegset.intersection(self.all_questions_seg_set[fields[5]]))
#                 if intersect == len(qsegset) and intersect == len(self.all_questions_seg_set[fields[5]]):
#                     if fields[3] == self.all_questions_seg[fields[5]]:
#                         score = 0.99
#                     else:
#                         score = 0.9
                for i in range(len(topK)):
                    if topK[i][0] < score:
                        topK[i+1:] = topK[i:-1]
                        topK[i] = (score,fields[5])
                        break
        return topK
    
    def decode(self, model_prefix, in_path, out_path, word_vec_path, mode, out_json_path=None, dump_prob_path=None):
    #     model_prefix = args.model_prefix
    #     in_path = args.in_path
    #     out_path = args.out_path
    #     word_vec_path = args.word_vec_path
    #     mode = args.mode
    #     out_json_path = None
    #     dump_prob_path = None
        
        # load the configuration file
        print('Loading configurations.')
        FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")
        print(FLAGS)
    
        with_POS=False
        if hasattr(FLAGS, 'with_POS'): with_POS = FLAGS.with_POS
        with_NER=False
        if hasattr(FLAGS, 'with_NER'): with_NER = FLAGS.with_NER
        wo_char = False
        if hasattr(FLAGS, 'wo_char'): wo_char = FLAGS.wo_char
    
        wo_left_match = False
        if hasattr(FLAGS, 'wo_left_match'): wo_left_match = FLAGS.wo_left_match
    
        wo_right_match = False
        if hasattr(FLAGS, 'wo_right_match'): wo_right_match = FLAGS.wo_right_match
    
        wo_full_match = False
        if hasattr(FLAGS, 'wo_full_match'): wo_full_match = FLAGS.wo_full_match
    
        wo_maxpool_match = False
        if hasattr(FLAGS, 'wo_maxpool_match'): wo_maxpool_match = FLAGS.wo_maxpool_match
    
        wo_attentive_match = False
        if hasattr(FLAGS, 'wo_attentive_match'): wo_attentive_match = FLAGS.wo_attentive_match
    
        wo_max_attentive_match = False
        if hasattr(FLAGS, 'wo_max_attentive_match'): wo_max_attentive_match = FLAGS.wo_max_attentive_match
    
    
        # load vocabs
        print('Loading vocabs.')
        word_vocab = Vocab(word_vec_path, fileformat='txt3')
        label_vocab = Vocab(model_prefix + ".label_vocab", fileformat='txt2')
        print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
        print('label_vocab: {}'.format(label_vocab.word_vecs.shape))
        num_classes = label_vocab.size()
        
        POS_vocab = None
        NER_vocab = None
        char_vocab = None
        if with_POS: POS_vocab = Vocab(model_prefix + ".POS_vocab", fileformat='txt2')
        if with_NER: NER_vocab = Vocab(model_prefix + ".NER_vocab", fileformat='txt2')
        char_vocab = Vocab(model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
        
        print('Build SentenceMatchDataStream ... ')
        testDataStream = SentenceMatchTrainer.SentenceMatchDataStream(in_path, word_vocab=word_vocab, char_vocab=char_vocab, 
                                                  POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, 
                                                  batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=True, 
                                                  max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length)
        print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
        print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))
    
        if wo_char: char_vocab = None
    
        init_scale = 0.01
        best_path = model_prefix + ".best.model"
        print('Decoding on the test set:')
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-init_scale, init_scale)
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab, 
                     dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                     lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim, 
                     aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False, MP_dim=FLAGS.MP_dim, 
                     context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num, 
                     fix_word_vec=FLAGS.fix_word_vec,with_filter_layer=FLAGS.with_filter_layer, with_highway=FLAGS.with_highway,
                     word_level_MP_dim=FLAGS.word_level_MP_dim,
                     with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
                     highway_layer_num=FLAGS.highway_layer_num, with_lex_decomposition=FLAGS.with_lex_decomposition, 
                     lex_decompsition_dim=FLAGS.lex_decompsition_dim, with_char=(not FLAGS.wo_char),
                     with_left_match=(not FLAGS.wo_left_match), with_right_match=(not FLAGS.wo_right_match),
                     with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match), 
                     with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match))
    
            # remove word _embedding
            vars_ = {}
            for var in tf.all_variables():
                if "word_embedding" in var.name: continue
                if not var.name.startswith("Model"): continue
                vars_[var.name.split(":")[0]] = var
            saver = tf.train.Saver(vars_)
                    
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            step = 0
            best_path = best_path.replace('//', '/')
            saver.restore(sess, best_path)
    
            accuracy = SentenceMatchTrainer.evaluate(testDataStream, valid_graph, sess, outpath=out_path, label_vocab=label_vocab,mode=mode,
                                                     char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab)
#             print("Accuracy for test set is %.2f" % accuracy)
