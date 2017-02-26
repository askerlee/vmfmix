import sys
import pdb
import os
import getopt
from corpusLoader import *
from utils import *
from vmfmix import vmfMix

config = dict(  unigramFilename = "top1grams-wiki.txt",
                wordvecFilename = "25000-180000-500-BLK-8.0.vec",
                #wordvecFilename = "embeddings/wiki/word2vec.vec",
                K = 100,
                # for separate category training, each category has 10 topics, totalling 200
                sepK_20news = 15,
                sepK_reuters = 12,
                # set it to 0 to disable the removal of very small topics
                topTopicMassFracThres = 0.05,
                topW = 12,
                # when topTopicMassFracPrintThres = 0, print all topics
                topTopicMassFracPrintThres = 0,
                alpha = 0.1,
                MAX_EM_ITERS = 50,
                TDiff_tolerance = 2e-3,
                max_kappa = 50,
                remove_stop = True,
                verbose = 0,
                seed = 0,
                printTopics_iterNum = 10
            )

def usage():
    print """Usage: wordclust.py -s                corpus_name set_name(s)
                   -i topic_vec_file corpus_name set_name(s)
                   [ -w ]            corpus_name set_name(s)
                   (Optional) -t max_iter_num ...
  corpus_name: '20news' or  'reuters'
  set_name(s): 'train', 'test' or 'train,test' (will save in separate files)
  -s:          Train on separate categories
  -i:          Do inference on a corpus given a topic vec file
  -w:          Dump words only (no inference of topics)
  -t:          Specify the maximum number of iterations"""

def loadwordvecs(unigramFilename, wordvecFilename):
    vocab_dict = loadUnigramFile(unigramFilename)
    
    embedding_npyfile = wordvecFilename + ".npy"
    if os.path.isfile(embedding_npyfile):
        print "Load embeddings from npy file '%s'" %embedding_npyfile
        embedding_arrays = np.load(embedding_npyfile)
        V, vocab, word2ID, skippedWords_whatever = embedding_arrays
    else:
        V, vocab, word2ID, skippedWords_whatever = load_embeddings(wordvecFilename)
        embedding_arrays = np.array( [ V, vocab, word2ID, skippedWords_whatever ] )
        print "Save embeddings to npy file '%s'" %embedding_npyfile
        np.save( embedding_npyfile, embedding_arrays )
        
    V = normalizeF(V)
    return V, word2ID

def docSentences2embeddings(docs_wordsInSentences, V, word2ID, remove_stop=True):
    X = []
    Words = []
    Freqs = []
    docs_idx = []

    word2freq = {}
    countedWC = 0
    outvocWC = 0
    stopwordWC = 0

    for i, wordsInSentences in enumerate(docs_wordsInSentences):
        Xi = []
        doc_words = []
        doc_freqs = []
        doc_word2freq = {}
        
        for sentence in wordsInSentences:
            for w in sentence:
                w = w.lower()
                if remove_stop and w in stopwordDict:
                    stopwordWC += 1
                    continue

                if w in word2freq:
                    word2freq[w] += 1
                else:
                    word2freq[w] = 1
                    
                if w in doc_word2freq:
                    doc_word2freq[w] += 1
                    continue
                if w in word2ID:
                    wid = word2ID[w]
                    Xi.append(V[wid])
                    doc_words.append(w)
                    doc_word2freq[w] = 1
                    countedWC += 1
                else:
                    outvocWC += 1

        # skip empty documents
        if len(Xi) > 0:
            X.append( np.array(Xi) )
            Words.append(doc_words)
            doc_wordfreqs = [ doc_word2freq[w] for w in doc_words ]
            Freqs.append( np.array(doc_wordfreqs) )
            docs_idx.append(i)
            
    # out0 prints both to screen and to log file, regardless of the verbose level
    global out0, out1

    out0( "%d docs scanned, %d kept. %d words kept, %d unique. %d stop words, %d out voc" %( len(docs_wordsInSentences),
                                                        len(X), countedWC, len(word2freq), stopwordWC, outvocWC ) )

    word_freqs = sorted( word2freq.items(), key=lambda kv: kv[1], reverse=True )
    out1("Top words:")
    line = ""
    for w, freq in word_freqs[:30]:
        line += "%s: %d " %( w, freq )
    out1(line)
    return docs_idx, X, Freqs, Words, word2freq
            
corpusName = None
corpus2loader = { '20news': load_20news, 'reuters': load_reuters }
    
subsetNames = [ ]
topic_vec_file = None
MAX_ITERS = -1
MAX_TopicProp_ITERS = 1
onlyDumpWords = False
separateCatTraining = False
onlyInferTopicProp = False
topicTraitStr = ""
onlyGetOriginalText = False

try:
    opts, args = getopt.getopt( sys.argv[1:], "i:t:wso" )

    if len(args) == 0:
        raise getopt.GetoptError("Not enough free arguments")
    corpusName = args[0]
    if len(args) == 2:
        subsetNames = args[1].split(",")
    if len(args) > 2:
        raise getopt.GetoptError("Too many free arguments")

    for opt, arg in opts:
        if opt == '-i':
            onlyInferTopicProp = True
            topic_vec_file = arg
        if opt == '-t':
            MAX_ITERS = int(arg)
        if opt == '-w':
            onlyDumpWords = True
        if opt == '-s':
            separateCatTraining = True
        if opt == '-o':
            onlyGetOriginalText = True
            
except getopt.GetoptError, e:
    print e.msg
    usage()
    sys.exit(2)

if not onlyGetOriginalText:
# The leading 'all-mapping' is only to get word mappings from the original IDs in 
# the embedding file to a compact word ID list, to speed up computation of sLDA
# The mapping has to be done on 'all' to include all words in train and test sets
    subsetNames = [ 'all-mapping' ] + subsetNames
    V, word2ID = loadwordvecs(config['unigramFilename'], config['wordvecFilename'])
else:
    V, word2ID = None, None
        
if MAX_ITERS > 0:
    if onlyInferTopicProp:
        MAX_TopicProp_ITERS = MAX_ITERS
    else:
        config['MAX_EM_ITERS'] = MAX_ITERS

loader = corpus2loader[corpusName]
word2compactId = {}
compactId_words = []
hasIdMapping = False

if onlyInferTopicProp:
    topicfile_trunk = topic_vec_file.split(".")[0]
    topicTraits = topicfile_trunk.split("-")[3:]
    topicTraitStr = "-".join(topicTraits)
    T_kappa = load_matrix_from_text( topic_vec_file, "topic" )
    # first dimension of the saved matrix is kappa, the remaining are T
    kappa = T_kappa[:,0]
    T = T_kappa[:,1:]
    config['K'] = T.shape[0]
    
config['logfilename'] = corpusName
vmfmixer = vmfMix(**config)
out0 = vmfmixer.genOutputter(0)
out1 = vmfmixer.genOutputter(1)

for si, subsetName in enumerate(subsetNames):       
    print "Process subset '%s':" %subsetName
    if subsetName == 'all-mapping':
        subsetName = 'all'
        onlyGetWidMapping = True
    else:
        onlyGetWidMapping = False
        
    subsetDocNum, orig_docs_words, orig_docs_name, orig_docs_cat, cats_docsWords, \
            cats_docNames, category_names = loader(subsetName)
    catNum = len(category_names)
    basename = "%s-%s-%d" %( corpusName, subsetName, subsetDocNum )

    # dump original words (without filtering)
    orig_filename = "%s.orig.txt" %basename
    ORIG = open( orig_filename, "w" )
    for wordsInSentences in orig_docs_words:
        for sentence in wordsInSentences:
            for i,w in enumerate(sentence):
                sentence[i] = w.lower()        
                ORIG.write( "%s " %sentence[i] )
        ORIG.write("\n")
    ORIG.close()
    print "%d original docs saved in '%s'" %( subsetDocNum, orig_filename )

    if onlyGetOriginalText:
        continue
        
    docs_idx, X, Freqs, Words, word2freq = docSentences2embeddings( orig_docs_words, V, word2ID )
    vmfmixer.setX( corpusName, X, Freqs, Words )
    docs_name = [ orig_docs_name[i] for i in docs_idx ]
    docs_cat = [ orig_docs_cat[i] for i in docs_idx ]
    readDocNum = len(docs_idx)
    out0( "%d docs left after filtering empty docs" %(readDocNum) )

    # executed when subsetName == 'all-mapping'
    if onlyGetWidMapping:
        sorted_words = sorted( word2freq.keys() )
        uniq_wid_num = len(sorted_words)
        for i, w in enumerate(sorted_words):
            word2compactId[w] = i + 1
            compactId_words.append(w)
            
        hasIdMapping = True
        onlyGetWidMapping = False
        print "Word mapping created: %s -> %d" %( sorted_words[-1], uniq_wid_num )
        id2word_filename = "%s.id2word.txt" %basename
        ID2WORD = open( id2word_filename, "w" )
        for i in xrange(uniq_wid_num):
            ID2WORD.write( "%d\t%s\n" %( i, compactId_words[i] ) )
        ID2WORD.close()
        continue
                    
    # load topics from a file, infer the topic proportions, and save the proportions
    if onlyInferTopicProp:
        ni_k, Pi = vmfmixer.inferTopicProps(T, kappa, MAX_TopicProp_ITERS)
        # dump the topic proportions in my own matrix format
        save_matrix_as_text( basename + "-%s-i%d.vmftopic.prop" %(topicTraitStr, MAX_TopicProp_ITERS), 
                                "topic proportion", ni_k, docs_cat, docs_name, colSep="\t" )
        
        # dump the topic proportions into SVMTOPIC_PROP in libsvm/svmlight format
        # dump the mix of word freqs and topic proportions into SVMTOPIC_BOW in libsvm/svmlight format   
        svmtopicprop_filename = "%s.svm-vmftopicprop.txt" %basename
        # topic props + weighted sum of topic vectors
        svmtopicbow_filename = "%s.svm-vmftopicbow.txt" %basename
        svmtopic_wvavg_filename = "%s.svm-vmftopic-wvavg.txt" %basename
        
        SVMTOPIC_PROP = open( svmtopicprop_filename, "w" )
        SVMTOPIC_BOW = open( svmtopicbow_filename, "w" )
        SVMTOPIC_WVAVG = open( svmtopic_wvavg_filename, "w" )
        
        wordvec_avg = np.zeros( vmfmixer.D )
        
        for i in xrange(readDocNum):
            words = Words[i]
            cwid2freq = {}
            for j,w in enumerate(words):
                cwid = word2compactId[w]
                cwid2freq[cwid] = Freqs[i][j]
                wordvec_avg += X[i][j] * Freqs[i][j]
                    
            catID = docs_cat[i]
            sorted_cwids = sorted( cwid2freq.keys() )
            
            SVMTOPIC_PROP.write( "%d" %(catID+1) )
            SVMTOPIC_BOW.write( "%d" %(catID+1) )
            SVMTOPIC_WVAVG.write( "%d" %(catID+1) )
            
            for k in xrange(vmfmixer.K):
                SVMTOPIC_PROP.write( " %d:%.3f" %( k+1, ni_k[i][k] ) )
                SVMTOPIC_BOW.write( " %d:%.3f" %( k+1, ni_k[i][k] ) )
                SVMTOPIC_WVAVG.write( " %d:%.3f" %( k+1, ni_k[i][k] ) )
            
            for cwid in sorted_cwids:
                # first K indices are reserved for topic features, so add vmfmixer.K here
                SVMTOPIC_BOW.write( " %d:%d" %( cwid + vmfmixer.K, cwid2freq[cwid] ) )
            
            wordvec_avg /= vmfmixer.L[i]
            for d in xrange(vmfmixer.D):
                SVMTOPIC_WVAVG.write( " %d:%.3f" %( d + 1 + vmfmixer.K, wordvec_avg[d] ) )
                
            SVMTOPIC_PROP.write("\n")    
            SVMTOPIC_BOW.write("\n")
            SVMTOPIC_WVAVG.write("\n")
            
        SVMTOPIC_PROP.close()
        SVMTOPIC_BOW.close()
        SVMTOPIC_WVAVG.close()
        
        print "%d docs saved in '%s' in svm vmfTopicProp format" %( readDocNum, svmtopicprop_filename )
        print "%d docs saved in '%s' in svm vmfTopicProp-BOW format" %( readDocNum, svmtopicbow_filename )
        print "%d docs saved in '%s' in svm vmfTopicProp-WordvecAvg format" %( readDocNum, svmtopic_wvavg_filename )
        
    # infer topics from docs, and save topics and their proportions in each doc
    else:
        if not separateCatTraining:
            best_last_Tkappa, ni_k = vmfmixer.inference()

            best_it, best_T, best_kappa, best_varLB = best_last_Tkappa[0]
            last_it, last_T, last_kappa, last_varLB = best_last_Tkappa[1]
            
            best_Tkappa = np.concatenate( (best_kappa[:,None], best_T), axis=1 )
            last_Tkappa = np.concatenate( (last_kappa[:,None], last_T), axis=1 )
                
            save_matrix_as_text( basename + "-em%d-best.vmftopic.vec" %best_it, "best T/Kappa's", best_Tkappa  )
            save_matrix_as_text( basename + "-em%d-last.vmftopic.vec" %last_it, "last T/Kappa's", last_Tkappa  )
                
            save_matrix_as_text( basename + "-em%d.vmftopic.prop" %config['MAX_EM_ITERS'], "topic proportion", ni_k, docs_cat, docs_name, colSep="\t" )

        else:
            # infer topics for each category, combine them and save in one file
            if corpusName == "20news":
                vmfmixer.K = config['sepK_20news']
            else:
                vmfmixer.K = config['sepK_reuters']
                
            best_Tkappa = []
            last_Tkappa = []
            slim_Tkappa = []
            totalDocNum = 0
            #pdb.set_trace()
            
            for catID in xrange(catNum):
                out0("")
                out0( "Inference on category %d:" %( catID+1 ) )
                cat_docs_idx, cat_X, cat_Freqs, cat_Words, cat_word2freq = docSentences2embeddings( cats_docsWords[catID], V, word2ID )
                vmfmixer.setX( corpusName + " cat-%d" %( catID+1 ), cat_X, cat_Freqs, cat_Words )
                totalDocNum += len(cat_X)
                cat_best_last_Tkappa, cat_ni_k = vmfmixer.inference()
                # these variables are category-specific
                best_it, best_T, best_kappa, best_varLB = cat_best_last_Tkappa[0]
                last_it, last_T, last_kappa, last_varLB = cat_best_last_Tkappa[1]
                    
                # normalize by the number of documents 
                cat_ni_k2 = vmfmixer.n__k / len(cat_X)
                
                cat_best_Tkappa = np.concatenate( (best_kappa[:,None], best_T), axis=1 )
                best_Tkappa.append( cat_best_Tkappa )
                cat_last_Tkappa = np.concatenate( (last_kappa[:,None], last_T), axis=1 )
                last_Tkappa.append( cat_last_Tkappa )
 
                sorted_tids = sorted( range(vmfmixer.K), key=lambda k: cat_ni_k2[k], reverse=True )
                out0("Topic normalized mass:")
                s = ""
                for tid in sorted_tids:
                    s += "%d: %.3f " %( tid, cat_ni_k2[tid] )
                out0(s)
                
                if config['topTopicMassFracThres'] > 0:
                    cat_mass_thres = np.sum(cat_ni_k2) / vmfmixer.K * config['topTopicMassFracThres']
                    out0( "Topic normalized mass thres: %.3f" %cat_mass_thres )
                    top_tids = []
                    for i,tid in enumerate(sorted_tids):
                        if cat_ni_k2[tid] <= cat_mass_thres:
                            break
                        top_tids.append(tid)
                        
                    out0( "Keep top %d topics:" %len(top_tids) )
                    s = ""
                    for tid in top_tids:
                        s += "%d: %.3f " %( tid, cat_ni_k2[tid] )
                    out0(s)
                    
                    slim_cat_T = last_T[top_tids]
                    slim_cat_kappa = last_kappa[top_tids]
                    slim_cat_Tkappa = np.concatenate( (slim_cat_kappa[:,None], slim_cat_T), axis=1 )
                    slim_Tkappa.append(slim_cat_Tkappa)
                
            out0( "Done inference on %d docs in %d categories" %(totalDocNum, catNum) )

            best_Tkappa = np.concatenate(best_Tkappa)
            last_Tkappa = np.concatenate(last_Tkappa)
            save_matrix_as_text( "%s-sep%d-em%d-best.vmftopic.vec" %( basename, 
                best_Tkappa.shape[0], config['MAX_EM_ITERS'] ), "best topics", best_Tkappa )
            save_matrix_as_text( "%s-sep%d-em%d-last.vmftopic.vec" %( basename, 
                last_Tkappa.shape[0], config['MAX_EM_ITERS'] ), "last topics", last_Tkappa )

            if config['topTopicMassFracThres'] > 0:
                slim_Tkappa = np.concatenate(slim_Tkappa)
                save_matrix_as_text( "%s-sep%d-em%d-slim.vmftopic.vec" %( basename, 
                    slim_Tkappa.shape[0], config['MAX_EM_ITERS'] ), "slim topics", slim_Tkappa )
            